# Run:
#   python -m sepmi.examples.05_select_topk_admm \
#     --cov /mnt/data/covariance_matrix.pickle \
#     --tgt /mnt/data/weight_benchmark.pickle \
#     --k 100 --out /mnt/data/selected_top100_admm.csv --solver normal_cholmod
#
# Problem:
#   min_w (w - w_tgt)^T Σ (w - w_tgt)
#   s.t.  w >= 0,  1^T w = 1,  ||w||_0 = K  (exact K)
#
# ADMM split: f(x)=0.5 x^T Σ x + (-Σw_tgt)^T x,  g(z)=I_{Δ_K}(z),  x=z
#   x-step: (Σ + ρ I) x = Σ w_tgt + ρ (z - u)         (SPD solve)
#   z-step: z = Proj_{Δ_K}( v )  with v = αx + (1-α)z + u
#          (exact-K variant: keep top-K by value, simplex-project those,
#           and if any become zero, swap in next best until K strictly > 0)

from __future__ import annotations
import argparse, math, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sepmi.admm_core import admm_solve, CholmodNormalSolverReusable

# --------------------- IO + alignment ---------------------
def _align_and_clean(cov, tgt):
    if isinstance(cov, np.ndarray):
        cov = pd.DataFrame(cov)
    if isinstance(tgt, pd.DataFrame):
        # use first column if DataFrame
        tgt = tgt.iloc[:, 0]
    if not isinstance(cov, pd.DataFrame) or not isinstance(tgt, pd.Series):
        raise TypeError("Expected cov: DataFrame (or ndarray), tgt: Series (or 1-col DataFrame).")

    # ensure square, labeled covariance
    if not cov.index.equals(cov.columns):
        if len(cov.index) != len(cov.columns):
            n = cov.shape[0]
            cov.index = [f"ASSET_{i}" for i in range(n)]
            cov.columns = cov.index

    # align on common labels (and reorder target to cov order)
    common = cov.index.intersection(tgt.index)
    if len(common) == 0:
        if len(tgt) == cov.shape[0]:
            tgt.index = cov.index
            common = cov.index
        else:
            raise ValueError("No overlapping tickers between covariance and target.")
    cov = cov.loc[common, common].copy()
    tgt = tgt.loc[common].astype(float).copy()

    # symmetrize + tiny jitter for SPD niceness
    M = cov.values.astype(float, copy=True)
    M = 0.5 * (M + M.T)
    eps = 1e-10 * np.trace(M) / max(1, M.shape[0])
    np.fill_diagonal(M, np.diag(M) + eps)
    cov.loc[:, :] = M
    return cov, tgt

# --------------------- projections / prox ---------------------
def _project_simplex(y: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Project y onto {x >= 0, sum x = s}. (Duchi–Shalev-Shwartz–Singer–Chandra, 2008)
    """
    if s <= 0:
        z = np.zeros_like(y)
        return z
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - s
    rho_idx = np.nonzero(u - cssv / (np.arange(1, y.size + 1)) > 0)[0]
    if rho_idx.size == 0:
        z = np.zeros_like(y)
        z[np.argmax(y)] = s
        return z
    rho = rho_idx[-1]
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(y - theta, 0.0)
    return w

def prox_exact_k_simplex(K: int):
    """
    Project v onto Δ_K = { z >= 0, 1^T z = 1, ||z||_0 = K }.
    Strategy:
      - Start with top-K by value (not abs).
      - Simplex-project on that set.
      - If any projected coord hits 0, replace it with the next best index
        from the remainder, and re-project, until exactly K are strictly > 0
        (or remainder is exhausted; then you'll get ≤K).
    Deterministic and fast when K << n.
    """
    def prox(v: np.ndarray, t: float) -> np.ndarray:
        n = v.size
        K_eff = min(K, n)
        order = np.argsort(v)[::-1]      # descending by value
        S = list(order[:K_eff])
        R = list(order[K_eff:])          # remainder, descending

        def project_on(S_idx):
            z = np.zeros_like(v)
            zS = _project_simplex(v[S_idx], s=1.0)   # nonneg + sum1
            z[S_idx] = zS
            return z, int((zS > 0).sum())

        z, nzpos = project_on(S)
        # bring in more names if some coordinates collapsed to zero
        while nzpos < K_eff and len(R) > 0:
            # indices in S whose z became (near) zero
            zero_in_S = [i for i in S if z[i] <= 0.0]
            if not zero_in_S:
                break
            take = min(len(zero_in_S), len(R))
            for j in range(take):
                # swap one zeroed index with next best from R
                idx_zero = zero_in_S[j]
                pos = S.index(idx_zero)
                S[pos] = R[j]
            R = R[take:]
            z, nzpos = project_on(S)
            if take == 0:
                break
        return z
    return prox

# --------------------- K schedule ---------------------
def build_k_schedule(n: int, K_final: int):
    """
    Build a descending K schedule, warm-starting each stage:
      Prefer [500, 400, 300, 200, 100] filtered to [>= K_final, <= n],
      and ensure the last stage equals K_final (even if not in the base list).
    """
    base = [100]
    sched = [k for k in base if (k <= n and k >= K_final)]
    if len(sched) == 0:
        sched = [min(n, K_final)]
    if sched[-1] != K_final:
        if K_final <= n:
            sched.append(K_final)
        else:
            sched.append(n)
    # ensure strictly descending & unique
    out = []
    for k in sched:
        if len(out) == 0 or k < out[-1]:
            out.append(k)
    return out

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True, help="Pickle or CSV with square covariance (index==columns).")
    ap.add_argument("--tgt", required=True, help="Pickle/CSV Series or 1-col DataFrame of benchmark weights.")
    ap.add_argument("-k", "--k", type=int, default=100, help="Exact K to end with.")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=600, help="Iterations per stage.")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--solver", default="normal_cholmod",
                    choices=["auto","normal","normal_cholmod","woodbury","kkt","kkt_petsc","kkt_mumps"])
    ap.add_argument("--out", default="/mnt/data/selected_topK_admm.csv")
    args = ap.parse_args()

    # Load covariance (pickle or CSV)
    if args.cov.lower().endswith(".csv"):
        cov = pd.read_csv(args.cov, index_col=0)
    else:
        cov = pd.read_pickle(args.cov)

    # Load target weights (pickle or CSV)
    if args.tgt.lower().endswith(".csv"):
        tgt_df = pd.read_csv(args.tgt, index_col=0)
        tgt = tgt_df.iloc[:, 0] if isinstance(tgt_df, pd.DataFrame) else tgt_df
    else:
        tgt = pd.read_pickle(args.tgt)

    cov, tgt = _align_and_clean(cov, tgt)
    Sigma = cov.values
    tickers = cov.index.to_numpy()
    n = Sigma.shape[0]
    K_final = min(args.k, n)

    # ADMM data: P=Σ, q = -Σ w_tgt (since 0.5 x^T Σ x - (Σ w_tgt)^T x up to const)
    P = sp.csc_matrix(Sigma)
    q = -(Sigma @ tgt.values)

    # Build K schedule and warm-start progressively
    k_sched = build_k_schedule(n, K_final)

    # Optionally start from a simple K=all simplex projection of w_tgt (already nonneg?)
    z_warm = _project_simplex(tgt.values.copy(), s=1.0)
    u_warm = np.zeros_like(z_warm, dtype=float)
    x_warm = z_warm.copy()

    print(f"[info] n={n}, target K={K_final}, schedule={k_sched}")

    t0_all = time.perf_counter()
    for i, K in enumerate(k_sched, 1):
        prox_z = prox_exact_k_simplex(K)
        print(f"\n--- Stage {i}/{len(k_sched)} : K={K} ---")
        lin = None
        if args.solver == "normal_cholmod":
            lin = CholmodNormalSolverReusable(P, rho=args.rho)
        res = admm_solve(
            P, q,
            A=None, b=None,                 # equality handled by z-prox (sum=1)
            prox_z=prox_z,
            rho=args.rho, beta=0.0, alpha=args.alpha,
            max_iter=args.iters, rtol=args.rtol, atol=args.atol,
            verbose=False, profile=True, profile_interval=100, print_header=False,
            solver=args.solver,
            warm={"x": x_warm, "z": z_warm, "u": u_warm},
            lin_solver=lin
        )
        z = np.asarray(res["z"], float)
        x_warm = z.copy()
        z_warm = z.copy()
        u_warm = np.zeros_like(z)  # reset dual for the next stage tends to work well

        # Stage report
        diff = z - tgt.values
        te = float(np.sqrt(diff @ (Sigma @ diff)) * 100.0)
        nnz = int((z > 0).sum())
        print(f"  iters={res['iters']}  nnz={nnz}  sum(w)={z.sum():.10f}  TE={te:.6f}% "
              f"(factor={res['stats']['factor_time']:.3f}s total={res['stats']['total_time']:.3f}s)")

    t_all = time.perf_counter() - t0_all
    w = z_warm  # final weights from last stage (exact-K nonneg simplex)

    # Save selection
    sel = (
        pd.DataFrame({"ticker": tickers, "weight": w, "absw": np.abs(w)})
        .query("absw > 0").drop(columns=["absw"])
        .sort_values("weight", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
    sel.to_csv(args.out, index=False)

    # Final report
    diff = w - tgt.values
    te = float(np.sqrt(diff @ (Sigma @ diff)) * 100.0)
    print(f"\n=== Final (K={K_final}) ===")
    print(f"Selected names: {int((w>0).sum())}  sum(w)={w.sum():.10f}  min(w)={w.min():.3e}")
    print(f"Tracking Error: {te:.6f}%   total wall time: {t_all:.3f}s")
    print(f"→ wrote {args.out}")

if __name__ == "__main__":
    main()
