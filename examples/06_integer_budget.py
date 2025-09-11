# sepmi/examples/06_integer_budget_admm.py
from __future__ import annotations
import argparse, numpy as np, pandas as pd, scipy.sparse as sp, time
from sepmi.strategies import ADMMStrategy
from sepmi.io import load_series_any, align_prices_weights
from sepmi.prox_catalog import prox_abs, prox_s_band, prox_d_grid_min_dollars
from sepmi.polish import polish_to_band

# ---------- equalities: A x = b with x = [d (n); y (n); s (1)] ----------
def build_equalities(prices: np.ndarray, target_spend: np.ndarray):
    """
    y - d = -t  (n rows)
    1^T d - s = 0  (1 row)
    """
    n = prices.size
    rows, cols, data = [], [], []
    b = np.empty(n + 1, dtype=float)

    # y - d = -t
    for i in range(n):
        # -d_i
        rows.append(i); cols.append(i);       data.append(-1.0)
        # +y_i
        rows.append(i); cols.append(n + i);   data.append(+1.0)
        b[i] = -target_spend[i]

    # 1^T d - s = 0
    row = n
    for i in range(n):
        rows.append(row); cols.append(i);     data.append(1.0)   # d block
    rows.append(row); cols.append(2*n);       data.append(-1.0)  # -s
    b[n] = 0.0

    A = sp.coo_matrix((data, (rows, cols)), shape=(n + 1, 2*n + 1)).tocsc()
    return A, b

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--tgt",     required=True)
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--max_residual", type=float, default=500.0)
    ap.add_argument("--lam1", type=float, default=1.0)
    ap.add_argument("--min_dollars", type=float, default=0.0)
    ap.add_argument("--decimals", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--solver", default="normal_cholmod",
        choices=["auto","normal","normal_cholmod","woodbury","kkt","kkt_petsc","kkt_mumps"])
    ap.add_argument("--out", default="/mnt/data/units_solution.csv")
    args = ap.parse_args()

    # load + align
    prices = load_series_any(args.prices)
    w_tgt  = load_series_any(args.tgt)
    prices, w_tgt = align_prices_weights(prices, w_tgt)

    tickers = prices.index.to_numpy()
    p = prices.values.astype(float)
    n = p.size
    C = float(args.capital)
    R = float(args.max_residual)
    lam1 = float(args.lam1)
    min_dollars = max(0.0, float(args.min_dollars))
    decimals = int(args.decimals)

    t = (w_tgt.values.astype(float) * C)  # per-name target dollars
    A, b = build_equalities(p, t)         # y - d = -t, 1^T d - s = 0

    # No quadratic term
    Pquad = sp.csc_matrix((2*n + 1, 2*n + 1), dtype=float)
    qvec  = np.zeros(2*n + 1, dtype=float)

    # prox over x = [d, y, s]
    def prox_z(v: np.ndarray, tstep: float) -> np.ndarray:
        vd = v[:n]
        vy = v[n:2*n]
        vs = v[2*n]

        zd = prox_d_grid_min_dollars(vd, tstep, prices=p, min_dollars=min_dollars, decimals=decimals)
        zy = prox_abs(vy, tstep, weight=lam1)
        zs = prox_s_band(vs, tstep, C=C, R=R)

        out = np.empty_like(v)
        out[:n] = zd
        out[n:2*n] = zy
        out[2*n] = zs
        return out

    beta_eq = 1e4

    start_time = time.time()
    strat = ADMMStrategy()
    res = strat.solve(
        Pquad, qvec,
        A=A, b=b,
        prox_z=prox_z,
        rho=args.rho, beta=beta_eq, alpha=args.alpha,
        max_iter=args.iters, rtol=args.rtol, atol=args.atol,
        verbose=False, lin_solver=None
    )
    solve_time = time.time() - start_time

    z = np.asarray(res["z"], float)
    d = z[:n]
    y = z[n:2*n]
    s = float(z[2*n])

    d = polish_to_band(d, p, t, C, R, min_dollars, decimals)
    y = d - t
    s = float(d.sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(p > 0, d / p, 0.0)

    spend = d
    total_spend = float(spend.sum())
    residual = total_spend - C
    band_ok = (C - R - 1e-9) <= total_spend <= (C + 1e-9)

    out_df = (pd.DataFrame({
        "ticker": tickers,
        "price": p,
        "units": np.round(u, decimals),
        "spend": spend,
        "target_spend": t,
        "abs_dev": np.abs(spend - t),
    }).loc[lambda d_: d_.units > 0]
      .sort_values("spend", ascending=False)
      .reset_index(drop=True))
    out_df.to_csv(args.out, index=False)

    print(f"n={n}  capital={C:,.2f}  sum(spend)={total_spend:,.2f}  residual={residual:,.2f}  band_ok={band_ok}")
    print(f"min_dollars={min_dollars:.2f}  decimals={decimals}  L1(dev)={float(np.abs(y).sum()):,.2f}")
    print(f"iters={res['iters']}  pri={res['pri_res']:.2e}  dua={res['dual_res']:.2e}  eq={res['eq_res']:.2e}")
    print(f"solve_time={solve_time:.3f}s")
    print(f"→ wrote {args.out}")

if __name__ == "__main__":
    main()
