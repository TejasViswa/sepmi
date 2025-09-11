# examples/10_rtop_long_only.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Any, List

from sepmi.strategies import ADMMStrategy
from sepmi.prox_catalog import prox_exact_k_simplex, project_simplex


def _align_cov_and_target(cov: pd.DataFrame, tgt_like) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(tgt_like, pd.DataFrame):
        if tgt_like.shape[1] != 1:
            raise ValueError("target must be Series or 1-col DataFrame.")
        tgt = tgt_like.iloc[:, 0]
    else:
        tgt = pd.Series(tgt_like)
    # align by index; fill 0 for missing tickers in target
    tgt = tgt.reindex(cov.index).fillna(0.0).astype(float)
    return cov, tgt


def _build_k_schedule(n: int, K_final: int) -> List[int]:
    # minimal, descending schedule; append K_final if needed
    base = [100]
    sched = [k for k in base if (k <= n and k >= K_final)]
    if len(sched) == 0:
        sched = [min(n, K_final)]
    if sched[-1] != K_final:
        sched.append(min(K_final, n))
    out = []
    for k in sched:
        if len(out) == 0 or k < out[-1]:
            out.append(k)
    return out


def RTOP_long_only(
    calculation_data: Dict[str, Any],
    kappa: int = 100,
    rho: float = 1.0,
    alpha: float = 1.6,
    iters_per_stage: int = 600,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Long-only exact-K selection, same setup as select_topk:
      min_w  (w - w_tgt)^T Σ (w - w_tgt)
      s.t.   w >= 0, 1^T w = 1, ||w||_0 = K

    Inputs in calculation_data:
      - 'cov_df': square covariance DataFrame (index==columns)
      - 'target_weights' or 'benchmark_weights': Series/1-col DataFrame/array
      - optional 'index_tickers' (defaults to cov_df.index)
    """
    t0 = time.perf_counter()

    cov_df = calculation_data["cov_df"]
    if not isinstance(cov_df, pd.DataFrame):
        raise ValueError("cov_df must be a pandas DataFrame with index==columns.")
    tgt_like = calculation_data.get("benchmark_weights", calculation_data.get("target_weights"))
    if tgt_like is None:
        raise ValueError("Provide 'benchmark_weights' or 'target_weights' in calculation_data.")

    cov_df, tgt = _align_cov_and_target(cov_df, tgt_like)
    Sigma = cov_df.values.astype(float)
    tickers = cov_df.index.to_numpy()
    n = Sigma.shape[0]
    K_final = int(min(max(1, kappa), n))

    # ADMM quadratic form (same as example):
    #   0.5 x^T P x + q^T x with P = Σ, q = -(Σ w_tgt)
    P = sp.csc_matrix(Sigma)
    q = -(Sigma @ tgt.values)

    # K schedule and warm-start
    k_sched = _build_k_schedule(n, K_final)
    z_warm = project_simplex(tgt.values.copy(), s=1.0)
    u_warm = np.zeros_like(z_warm, dtype=float)
    x_warm = z_warm.copy()

    if verbose:
        print(f"[info] n={n}, target K={K_final}, schedule={k_sched}")

    strat = ADMMStrategy()
    for i, K in enumerate(k_sched, 1):
        if verbose:
            print(f"\n--- Stage {i}/{len(k_sched)} : K={K} ---")

        def prox_z(v: np.ndarray, tstep: float) -> np.ndarray:
            return prox_exact_k_simplex(v, tstep, K=K)

        res = strat.solve(
            P, q,
            A=None, b=None,
            prox_z=prox_z,
            rho=rho, beta=0.0, alpha=alpha,
            max_iter=iters_per_stage, rtol=rtol, atol=atol,
            verbose=False, warm={"x": x_warm, "z": z_warm, "u": u_warm},
            lin_solver=None,
        )
        z = np.asarray(res["z"], float)
        x_warm = z.copy()
        z_warm = z.copy()
        u_warm = np.zeros_like(z)

        # quick stage stats
        diff = z - tgt.values
        te = float(np.sqrt(diff @ (Sigma @ diff)) * 100.0)
        nnz = int((z > 0).sum())
        if verbose:
            print(f"  iters={res['iters']}  nnz={nnz}  sum(w)={z.sum():.10f}  TE={te:.6f}% "
                  f"(total={res['stats']['total_time']:.3f}s)")

    # Final result
    w = z_warm
    diff = w - tgt.values
    te = float(np.sqrt(diff @ (Sigma @ diff)) * 100.0)
    selected = np.where(w > 0)[0]
    out = {
        "weights": w,
        "tracking_error": te,
        "selected_assets": selected,
        "selected_tickers": [calculation_data.get("index_tickers", list(tickers))[i] for i in selected],
    }
    if verbose:
        tall = time.perf_counter() - t0
        print(f"\n=== Final (K={K_final}) ===")
        print(f"Selected names: {int((w>0).sum())}  sum(w)={w.sum():.10f}  min(w)={w.min():.3e}")
        print(f"Tracking Error: {te:.6f}%   total wall time: {tall:.3f}s")
    return out