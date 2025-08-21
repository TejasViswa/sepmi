#!/usr/bin/env python3
import numpy as np, pandas as pd, argparse
from pathlib import Path

def power_iteration_maxeig(A, iters=100, tol=1e-7, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    v = rng.standard_normal(n); v /= (np.linalg.norm(v) + 1e-12)
    lam_old = 0.0
    for _ in range(iters):
        w = A @ v
        nv = np.linalg.norm(w)
        if nv < 1e-18: return 0.0
        v = w / nv
        lam = float(v @ (A @ v))
        if abs(lam - lam_old) < tol * (1.0 + abs(lam)): break
        lam_old = lam
    return lam

def hard_threshold_topk(x, K):
    if K >= x.size: return x.copy()
    idx = np.argpartition(np.abs(x), -K)[-K:]
    out = np.zeros_like(x); out[idx] = x[idx]; return out

def solve_topk_iht(cov, w_tgt, K=100, max_iter=200, tol=1e-9, seed=0):
    Σ = cov.values.astype(float)
    n = Σ.shape[0]; K = min(K, n)
    lam_max = power_iteration_maxeig(Σ, iters=100, seed=seed)
    Lg = max(2.0*lam_max, 1e-8); eta = 1.0 / Lg
    w = np.zeros(n, float)
    for _ in range(max_iter):
        grad = 2.0 * (Σ @ (w - w_tgt.values))
        w_new = hard_threshold_topk(w - eta * grad, K)
        if np.linalg.norm(w_new - w) <= tol * (1.0 + np.linalg.norm(w)): w = w_new; break
        w = w_new
    S = np.flatnonzero(w != 0.0)
    if S.size == 0:
        S = np.argpartition(np.abs(w_tgt.values), -K)[-K:]; S.sort()
    Sbar = np.setdiff1d(np.arange(n), S, assume_unique=True)
    Σ_SS = Σ[np.ix_(S, S)]
    RHS = Σ_SS @ w_tgt.values[S]
    if Sbar.size > 0: RHS = RHS + (Σ[np.ix_(S, Sbar)] @ w_tgt.values[Sbar])
    ridge = 1e-10 * np.trace(Σ_SS) / max(1, Σ_SS.shape[0]); 
    if ridge > 0: Σ_SS = Σ_SS + ridge * np.eye(Σ_SS.shape[0])
    try: w_S = np.linalg.solve(Σ_SS, RHS)
    except np.linalg.LinAlgError: w_S = np.linalg.lstsq(Σ_SS, RHS, rcond=None)[0]
    w_final = np.zeros(n, float); w_final[S] = w_S
    diff = w_final - w_tgt.values; obj = float(diff @ (Σ @ diff))
    return w_final, S, obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", default="/mnt/data/covariance_matrix.pickle")
    ap.add_argument("--tgt", default="/mnt/data/weight_benchmark.pickle")
    ap.add_argument("-k", "--topk", type=int, default=100)
    ap.add_argument("--out", default="/mnt/data/selected_top100.csv")
    args = ap.parse_args()

    cov = pd.read_pickle(args.cov)
    if isinstance(cov, np.ndarray): cov = pd.DataFrame(cov)
    tgt = pd.read_pickle(args.tgt)
    if isinstance(tgt, pd.DataFrame):
        tgt = tgt.iloc[:,0] if tgt.shape[1] >= 1 else pd.Series(dtype=float)

    common = cov.index.intersection(tgt.index) if isinstance(tgt, pd.Series) else cov.index
    if len(common) == 0 and isinstance(tgt, pd.Series) and len(tgt) == cov.shape[0]:
        tgt.index = cov.index; common = cov.index
    cov = cov.loc[common, common].copy()
    tgt = tgt.loc[common].astype(float).copy()

    M = cov.values; M = 0.5*(M+M.T); 
    eps = 1e-10*np.trace(M)/max(1,M.shape[0]); np.fill_diagonal(M, np.diag(M)+eps); cov.loc[:,:] = M

    w, S, obj = solve_topk_iht(cov, tgt, K=args.topk, max_iter=300, tol=1e-10, seed=42)
    tickers = cov.index.to_numpy()
    sel = pd.DataFrame({"ticker": tickers[S], "weight": w[S]})
    sel["abs_weight"] = np.abs(sel["weight"])
    sel = sel.sort_values("abs_weight", ascending=False).drop(columns=["abs_weight"]).reset_index(drop=True)
    sel.to_csv(args.out, index=False)
    print(f"Wrote {len(S)} weights to {args.out} (objective={obj:.6g}).")

if __name__ == "__main__":
    main()