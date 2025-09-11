# sepmi/examples/08_trade_opt_l2.py
# Run:
#   python -m sepmi.examples.08_trade_opt_l2 \
#     --prices /path/to/prices.pkl --tgt /path/to/weights.pkl \
#     --capital 1_000_000 --max_residual 500.0 \
#     --lam2 1.0 --min_dollars 0 --decimals 0 \
#     --solver normal_cholmod --out /tmp/trade_units_l2.csv
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sepmi.strategies import ADMMStrategy
from sepmi.io import load_series_any, align_prices_weights
from sepmi.prox_catalog import prox_s_band, prox_d_grid_min_dollars
from sepmi.polish import polish_to_band


def build_sum_equality(n: int):
    """
    Build A, b for one equality: 1^T d - s = 0 with x = [d (n), s (1)]
    A ∈ R^{1×(n+1)}, b ∈ R
    """
    rows, cols, data = [], [], []
    row = 0
    for i in range(n):
        rows.append(row); cols.append(i);   data.append(1.0)   # d block
    rows.append(row); cols.append(n);       data.append(-1.0)  # -s
    A = sp.coo_matrix((data, (rows, cols)), shape=(1, n + 1)).tocsc()
    b = np.array([0.0], dtype=float)
    return A, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--tgt",     required=True, help="Benchmark weights (sum to 1; will be normalized).")
    ap.add_argument("--capital", type=float, required=True, help="Total capital C in dollars.")
    ap.add_argument("--max_residual", type=float, default=500.0, help="Allow sum(d) ∈ [C-R, C].")
    ap.add_argument("--lam2", type=float, default=1.0, help="Weight for 0.5*lam2*||d - C*w_tgt||_2^2.")
    ap.add_argument("--min_dollars", type=float, default=0.0, help="Per-name minimum spend if nonzero.")
    ap.add_argument("--decimals", type=int, default=0, help="Lot size q=10^{-decimals}.")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--solver", default="normal_cholmod",
        choices=["auto","normal","normal_cholmod","woodbury","kkt","kkt_petsc","kkt_mumps"])
    ap.add_argument("--out", default="/tmp/trade_units_l2.csv")
    args = ap.parse_args()

    # Load + align
    prices = load_series_any(args.prices)
    w_tgt  = load_series_any(args.tgt)
    prices, w_tgt = align_prices_weights(prices, w_tgt)

    tickers = prices.index.to_numpy()
    p = prices.values.astype(float)
    n = p.size
    C = float(args.capital)
    R = float(args.max_residual)
    lam2 = float(args.lam2)
    min_dollars = max(0.0, float(args.min_dollars))
    decimals = int(args.decimals)

    # Target spend vector t = C * w_tgt, and x = [d (n), s (1)]
    t = (w_tgt.values.astype(float) * C)

    # Equality 1^T d - s = 0
    A, b = build_sum_equality(n)

    # Quadratic closeness on d: 0.5*lam2*||d - t||_2^2
    # => P = diag([lam2*I_n, 0]), q = [-lam2*t; 0]
    diagP = np.concatenate([np.full(n, lam2, dtype=float), np.array([0.0])])
    P = sp.diags(diagP, format="csc")
    q = np.concatenate([-lam2 * t, np.array([0.0])])

    # Separable g via prox: d in discrete grid, s in [C-R, C]
    def prox_z(v: np.ndarray, tstep: float) -> np.ndarray:
        vd = v[:n]
        vs = v[n]
        zd = prox_d_grid_min_dollars(vd, tstep, prices=p, min_dollars=min_dollars, decimals=decimals)
        zs = prox_s_band(vs, tstep, C=C, R=R)
        out = np.empty_like(v)
        out[:n] = zd
        out[n]  = zs
        return out

    # Solve
    strat = ADMMStrategy()
    t0 = time.time()
    res = strat.solve(
        P, q,
        A=A, b=b,
        prox_z=prox_z,
        rho=args.rho, beta=1e4, alpha=args.alpha,
        max_iter=args.iters, rtol=args.rtol, atol=args.atol,
        verbose=False, lin_solver=None
    )
    solve_time = time.time() - t0

    z = np.asarray(res.get("z", res.get("x")), float)
    d = z[:n]            # spend dollars
    s = float(z[n])      # total spend

    # Greedy polish to band [C-R, C] on discrete grid, minimizing |d - t|/$
    d = polish_to_band(d, p, t, C, R, min_dollars, decimals)
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

    l2 = float(np.linalg.norm(spend - t))
    l1 = float(np.abs(spend - t).sum())
    print(f"n={n}  capital={C:,.2f}  sum(spend)={total_spend:,.2f}  residual={residual:,.2f}  band_ok={band_ok}")
    print(f"L2(dev)={l2:,.2f}  L1(dev)={l1:,.2f}")
    print(f"iters={res['iters']}  pri={res['pri_res']:.2e}  dua={res['dual_res']:.2e}  eq={res['eq_res']:.2e}  time={solve_time:.3f}s")
    print(f"→ wrote {args.out}")


if __name__ == "__main__":
    main()