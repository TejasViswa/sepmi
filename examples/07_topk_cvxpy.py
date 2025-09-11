# Run:
#   python -m sepmi.examples.07_topk_cvxpy \
#     --cov /path/to/cov.pkl --tgt /path/to/weights.pkl \
#     --k 100 --out /tmp/selected_topK_cvxpy.csv --solver normal_cholmod

from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cvxpy as cp

from sepmi import solve_qp_into_cvxpy
from sepmi.io import align_covariance_and_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True, help="Pickle or CSV with square covariance (index==columns).")
    ap.add_argument("--tgt", required=True, help="Pickle/CSV Series or 1-col DataFrame of benchmark weights.")
    ap.add_argument("-k", "--k", type=int, default=100, help="Exact K to end with.")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=600, help="ADMM iterations.")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--solver", default="normal_cholmod",
                    choices=["auto","normal","normal_cholmod","woodbury","kkt","kkt_petsc","kkt_mumps"])
    ap.add_argument("--out", default="/tmp/selected_topK_cvxpy.csv")
    args = ap.parse_args()

    # Load inputs
    if args.cov.lower().endswith(".csv"):
        cov = pd.read_csv(args.cov, index_col=0)
    else:
        cov = pd.read_pickle(args.cov)

    if args.tgt.lower().endswith(".csv"):
        tgt_df = pd.read_csv(args.tgt, index_col=0)
        tgt = tgt_df.iloc[:, 0] if isinstance(tgt_df, pd.DataFrame) else tgt_df
    else:
        tgt = pd.read_pickle(args.tgt)

    cov, tgt = align_covariance_and_target(cov, tgt)
    Sigma = cov.values.astype(float)
    tickers = cov.index.to_numpy()
    n = Sigma.shape[0]
    K = min(args.k, n)

    # QP in SEPMI form: P=Σ, q = -Σ w_tgt (difference of means formulation)
    P = sp.csc_matrix(Sigma)
    q = -(Sigma @ tgt.values.astype(float))

    # cvxpy variable to receive SEPMI solution
    w = cp.Variable(n)

    # Separable g: exact-K nonnegative simplex over entire vector
    gspec = [{"g": "exact_k_simplex", "range": (0, n), "args": {"K": K}}]

    print(f"[info] n={n}, K={K}")
    t0 = time.perf_counter()
    res = solve_qp_into_cvxpy(
        w, P, q,
        A=None, b=None, gspec=gspec,
        solver=args.solver,
        rho=args.rho, beta=0.0, alpha=args.alpha,
        max_iter=args.iters, rtol=args.rtol, atol=args.atol,
    )
    t1 = time.perf_counter()

    w_val = np.asarray(w.value, float).ravel()
    diff = w_val - tgt.values.astype(float)
    te = float(np.sqrt(diff @ (Sigma @ diff)) * 100.0)
    nnz = int((w_val > 0).sum())
    print(f"iters={res['iters']}  nnz={nnz}  sum(w)={w_val.sum():.10f}  TE={te:.6f}%  time={t1-t0:.3f}s")

    # Persist selection
    out = (
        pd.DataFrame({"ticker": tickers, "weight": w_val, "absw": np.abs(w_val)})
        .query("absw > 0").drop(columns=["absw"])
        .sort_values("weight", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(args.out, index=False)
    print(f"→ wrote {args.out}")


if __name__ == "__main__":
    main() 