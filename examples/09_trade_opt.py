# sepmi/examples/06_trade_opt_sepmi_bridge.py
from __future__ import annotations
import time, argparse
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import Dict, Any

from sepmi.strategies import ADMMStrategy
from sepmi.prox_catalog import prox_s_band, prox_u_integer_min_dollars


# z ordering: [ lot_h (m); h (n); trade_units (m); s (1) ]
# Equalities:
#   (1) h - L @ lot_h = 0                 (n rows)
#   (2) lot_h - lot_init - P*trade_units = 0   (m rows),  P = diag(prices)
#   (3) prices^T * trade_units - s = 0    (1 row)
#   (4, optional) -sum_j (pnl_j/lot0_j) * lot_h_j + f_gains_trade_opt = max_realized_gains - sum_j pnl_j
#                  where f_gains_trade_opt >= 0 and is L1-minimized
def build_equalities(
    prices: np.ndarray,
    lot_to_ticker_matrix: np.ndarray,
    lot_init: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    p = np.asarray(prices, float).ravel()
    L = np.asarray(lot_to_ticker_matrix, float)
    l0 = np.asarray(lot_init, float).ravel()
    m = p.size
    n = int(L.shape[0])

    rows, cols, data = [], [], []
    b = np.zeros(n + m + 1, dtype=float)

    # (1) h - L lot_h = 0
    for i in range(n):
        rows.append(i); cols.append(m + i); data.append(1.0)  # +h_i
        nz = np.nonzero(L[i, :])[0]
        for j in nz:
            rows.append(i); cols.append(j); data.append(-float(L[i, j]))  # -L_{ij} * lot_h_j
        b[i] = 0.0

    # (2) lot_h - l0 - p * tu = 0
    row = n
    for j in range(m):
        rows.append(row); cols.append(j);             data.append(+1.0)        # lot_h_j
        rows.append(row); cols.append(m + n + j);     data.append(-float(p[j])) # -p_j * tu_j
        b[row] = float(l0[j])
        row += 1

    # (3) p^T tu - s = 0
    for j in range(m):
        rows.append(row); cols.append(m + n + j);     data.append(float(p[j]))
    rows.append(row); cols.append(m + n + m);         data.append(-1.0)
    b[row] = 0.0
    row += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(row, m + n + m + 1)).tocsc()
    return A, b


def solve_trade_opt_sepmi(env: Dict[str, Any], settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Quadratic + linear objective with equality coupling; discrete lot-dollar increments and cash-band via prox.

    env expects:
      prices (m,), lot_to_ticker_matrix (n x m), lot_init (m,),
      h_calc (n,), l_calc (m,) optional,
      c_avail (scalar), max_residual (scalar, default 250),
      h_dev_weight=lam_h (default 1.0), lot_dev_weight=lam_l (default 0.0), s_dev_weight=lam_s (default 0.0),
      min_alloc_per_asset (default 0.0), fractional_support (integer decimals, default 0),
      rho, alpha, iters, rtol, atol (optional).

      Optional for realized-gains cap:
      - max_realized_gains (scalar, set -1 to disable)
      - pnl_per_lot (m,), own_idx (indices or boolean mask of length m)
      - f_gains_weight (scalar L1 weight, default 1.0)
    """
    settings = settings or {}

    # Pull core inputs
    prices = env["prices"].value
    L = env["lot_to_ticker_matrix"]
    lot_init = env["lot_init"]
    c_avail = env["c_avail"]

    # Labels from provided tables
    lots_tbl: pd.DataFrame = env.get("new_lots_df", None)
    tick_tbl = env.get("market_values", None)

    # Targets
    h_calc = env["h"].value
    l_calc = env["lot_h"].value

    R = float(env.get("max_residual", 250.0))
    lam_h = float(env.get("h_dev_weight", 1.0))
    lam_l = float(env.get("lot_dev_weight", 1.0))
    lam_s = float(env.get("s_dev_weight", 1.0))
    min_dollars = float(env.get("min_alloc_per_asset", 0.0))

    # Optional realized-gains L1 weight
    lam_fg = float(env.get("f_gains_weight", 1.0))

    # Decimals (integer k for q=10^{-k}), assume scalar grid for lots (per-lot scalar supported by prox)
    decimals = np.min(env.get("fractional_support"))

    # Build equalities for z
    A, b = build_equalities(prices, L, lot_init)

    m = prices.size
    n = int(L.shape[0])
    N0 = m + n + m + 1  # base length without realized-gains slack

    # Optionally append realized-gains constraint row and slack variable (r >= 0)
    has_fg = False
    idx_fg = None
    max_realized_gains = float(env.get("max_realized_gains", -1.0))
    if max_realized_gains != -1.0:
        pnl_per_lot = env.get("pnl_per_lot", None)
        own_idx = env.get("own_idx", None)
        if pnl_per_lot is None or own_idx is None:
            raise KeyError("When using max_realized_gains, provide 'pnl_per_lot' and 'own_idx' in env.")
        pnl_per_lot = _as_float_array(pnl_per_lot)
        lot0 = _as_float_array(lot_init)
        if isinstance(own_idx, (list, tuple, np.ndarray, pd.Index)):
            own_idx = np.asarray(own_idx)
            if own_idx.dtype == bool:
                if own_idx.shape[0] != m:
                    raise ValueError("Boolean own_idx must have length m.")
                own_idx = np.where(own_idx)[0]
            else:
                own_idx = own_idx.astype(int)
        else:
            raise ValueError("own_idx must be an array-like of indices or boolean mask.")
        # Build coefficients for the new equality row
        coeff = -pnl_per_lot[own_idx] / lot0[own_idx]
        const_sum = float(pnl_per_lot[own_idx].sum())
        b_row = max_realized_gains - const_sum

        # Expand A to have a new column for slack and append the equality row
        A = sp.hstack([A, sp.csc_matrix((A.shape[0], 1))], format="csc")
        row_cols = list(own_idx.astype(int)) + [N0]  # lot_h indices and slack col
        row_data = list(map(float, coeff)) + [1.0]
        row_vec = sp.coo_matrix((row_data, ([0] * len(row_cols), row_cols)), shape=(1, N0 + 1))
        A = sp.vstack([A, row_vec], format="csc")
        b = np.concatenate([b, np.array([b_row], float)])
        has_fg = True
        idx_fg = N0  # index of the appended slack variable in z
        N = N0 + 1
    else:
        N = N0

    # Quadratic + linear objective: 0.5 z^T P z + q^T z
    # Blocks in order [l (m); h (n); u (m); s (1); f_gains (optional 1)]
    diagP = np.concatenate([
        np.full(m, lam_l, float),         # l
        np.full(n, lam_h, float),         # h
        np.zeros(m, float),               # u (no quad on u)
        np.array([lam_s], float),         # s
    ])
    if has_fg:
        diagP = np.concatenate([diagP, np.array([0.0], float)])
    P = sp.diags(diagP, format="csc")

    q_l = (-lam_l * l_calc) if (l_calc is not None and lam_l != 0.0) else np.zeros(m, float)
    q_h = (-lam_h * h_calc) if lam_h != 0.0 else np.zeros(n, float)
    q_u = np.zeros(m, float)
    q_s = np.array([-lam_s * c_avail], float)
    q = np.concatenate([q_l, q_h, q_u, q_s])
    if has_fg:
        q = np.concatenate([q, np.array([0.0], float)])

    # Prox on z = [l; h; u; s; (optional) f_gains]
    # - l: identity (no prox here; lot discretization handled by u)
    # - h: identity (all shaping handled by P, q)
    # - u: integer shares with per-ticker min dollars
    # - s: clamp to [c_avail - R, c_avail] via prox_s_band
    # - f_gains: positive soft-threshold for L1 + nonnegativity
    def prox_z(v: np.ndarray, tstep: float) -> np.ndarray:
        out = v.copy()

        v_l = v[:m]
        out[:m] = v_l

        # h block: identity
        # out[m:m + n] = v[m:m + n]

        # u block: integer shares with per-ticker min dollars
        v_u = v[m + n : m + n + m]
        out[m + n : m + n + m] = prox_u_integer_min_dollars(v_u, tstep, prices=prices, min_dollars=min_dollars)

        # s block: band to [c_avail - R, c_avail]
        vs = v[m + n + m]
        out[m + n + m] = prox_s_band(vs, tstep, C=c_avail, R=R)

        # f_gains slack: r >= 0 with L1 penalty
        if has_fg and idx_fg is not None:
            vf = v[idx_fg]
            # prox for lam_fg * ||r||_1 + I_{r>=0}
            out[idx_fg] = max(vf - tstep * lam_fg, 0.0)

        return out

    rho = float(env.get("rho", 1.0))
    alpha = float(env.get("alpha", 1.6))
    iters = int(env.get("iters", 1500))
    rtol = float(env.get("rtol", 1e-6))
    atol = float(env.get("atol", 1e-8))

    strat = ADMMStrategy()
    t0 = time.time()
    res = strat.solve(
        P, q,
        A=A, b=b,
        prox_z=prox_z,
        rho=rho, beta=1e4, alpha=alpha,
        max_iter=iters, rtol=rtol, atol=atol,
        verbose=False, lin_solver=None,
    )
    wall = time.time() - t0

    z = np.asarray(res.get("z", res.get("x")), float).ravel()
    lot_h = z[:m]
    h = z[m:m + n]
    trade_units = z[m + n:m + n + m]
    s = float(z[m + n + m])
    f_gains_trade_opt = float(z[idx_fg]) if has_fg else 0.0

    lots_df = lots_tbl.copy()
    lots_df["price"] = prices
    lots_df["lot_init"] = lot_init
    lots_df["trade_units"] = trade_units
    lots_df["l_calc"] = l_calc
    lots_df["lot_h"] = lot_h
    lots_df["lot_delta"] = lot_h - l_calc

    tickers_df = tick_tbl.copy()
    tickers_df["h_target"] = h_calc
    tickers_df["h_post"] = h
    tickers_df["h_abs_dev"] = np.abs(h - h_calc)

    # Optional realized gains value for diagnostics
    realized_gains = None
    if max_realized_gains != -1.0:
        # compute with provided own_idx and pnl_per_lot
        pnl_per_lot = _as_float_array(env["pnl_per_lot"]) if "pnl_per_lot" in env else None
        own_idx = env.get("own_idx", None)
        if pnl_per_lot is not None and own_idx is not None:
            if isinstance(own_idx, (list, tuple, np.ndarray, pd.Index)):
                own_idx = np.asarray(own_idx)
                if own_idx.dtype == bool:
                    own_idx = np.where(own_idx)[0]
                else:
                    own_idx = own_idx.astype(int)
            realized_gains = float(np.sum(pnl_per_lot[own_idx] * ( ( _as_float_array(lot_init)[own_idx] - lot_h[own_idx] ) / _as_float_array(lot_init)[own_idx] )))

    return dict(
        lot_h=lot_h, h=h, trade_units=trade_units, s=s,
        iters=int(res["iters"]), pri_res=float(res["pri_res"]),
        dual_res=float(res["dual_res"]), eq_res=float(res["eq_res"]),
        total_time=wall, lots_df=lots_df, tickers_df=tickers_df,
        f_gains_trade_opt=f_gains_trade_opt, realized_gains=realized_gains,
    )

def _as_float_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.values.astype(float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].values.astype(float)
        raise TypeError("Expected 1-col DataFrame for array-like field.")
    return np.asarray(x, float).ravel()


def _as_float_matrix(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.values.astype(float)
    return np.asarray(x, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Pickle file containing env dict (prices, L, lot_init, h_calc, etc.)")
    ap.add_argument("--out", required=True, help="Output CSV (lots). A second file with suffix _tickers.csv is also written.")
    # Optional overrides
    ap.add_argument("--h_dev_weight", type=float)
    ap.add_argument("--lot_dev_weight", type=float)
    ap.add_argument("--s_dev_weight", type=float)
    ap.add_argument("--max_residual", type=float)
    ap.add_argument("--min_alloc_per_asset", type=float)
    ap.add_argument("--fractional_support", type=int)
    ap.add_argument("--rho", type=float)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--rtol", type=float)
    ap.add_argument("--atol", type=float)
    args = ap.parse_args()

    # Load env dict from pickle
    env_obj = pd.read_pickle(args.params)
    if not isinstance(env_obj, dict):
        raise TypeError("Params pickle must contain a dict-like object.")
    env: Dict[str, Any] = dict(env_obj)

    # Optional overrides
    if args.h_dev_weight is not None: env["h_dev_weight"] = args.h_dev_weight
    if args.lot_dev_weight is not None: env["lot_dev_weight"] = args.lot_dev_weight
    if args.s_dev_weight is not None: env["s_dev_weight"] = args.s_dev_weight
    if args.max_residual is not None: env["max_residual"] = args.max_residual
    if args.min_alloc_per_asset is not None: env["min_alloc_per_asset"] = args.min_alloc_per_asset
    if args.fractional_support is not None: env["fractional_support"] = args.fractional_support
    if args.rho is not None: env["rho"] = args.rho
    if args.alpha is not None: env["alpha"] = args.alpha
    if args.iters is not None: env["iters"] = args.iters
    if args.rtol is not None: env["rtol"] = args.rtol
    if args.atol is not None: env["atol"] = args.atol

    # Solve
    res = solve_trade_opt_sepmi(env)

    lots_df = res["lots_df"]
    tickers_df = res["tickers_df"]

    out_lots = args.out
    out_tickers = out_lots.replace(".csv", "_tickers.csv") if out_lots.endswith(".csv") else (out_lots + "_tickers.csv")
    lots_df.to_csv(out_lots, index=True if lots_df.index.name is not None else False)
    tickers_df.to_csv(out_tickers, index=True if tickers_df.index.name is not None else False)

    m = env["lot_init"].shape[0]
    n = env["n_tickers"]
    print(f"m={m} lots  n={n} tickers  s={res['s']:,.2f}  iters={res['iters']}  "
          f"pri={res['pri_res']:.2e} dua={res['dual_res']:.2e} eq={res['eq_res']:.2e}  time={res['total_time']:.3f}s")
    print(f"→ wrote {out_lots} and {out_tickers}")


if __name__ == "__main__":
    main()