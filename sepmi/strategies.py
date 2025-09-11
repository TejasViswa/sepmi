# sepmi/strategies.py
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from typing import Callable, Optional, Dict, Any

Array = np.ndarray
Prox = Callable[[Array, float], Array]


def _csc(M):
    import scipy.sparse as sp
    return M if sp.isspmatrix_csc(M) else M.tocsc()


def _ensure_lin_solver(P, A, rho, beta, lin_solver):
    if lin_solver is not None:
        return lin_solver
    # default fallback
    from .linear_solvers import NormalEqSolver
    return NormalEqSolver(P, A, rho=rho, beta=beta)


class BaseStrategy:
    def solve(self, P, q, *, A=None, b=None, prox_z: Prox=None,
              rho: float=1.0, beta: float=5.0, alpha: float=1.6,
              max_iter: int=300, rtol: float=1e-4, atol: float=1e-6,
              verbose: bool=False, warm: Optional[Dict[str, Array]]=None,
              lin_solver: Optional[Any]=None) -> Dict[str, Any]:
        raise NotImplementedError


class ADMMStrategy(BaseStrategy):
    def solve(self, P, q, *, A=None, b=None, prox_z: Prox=None,
              rho: float=1.0, beta: float=5.0, alpha: float=1.6,
              max_iter: int=300, rtol: float=1e-4, atol: float=1e-6,
              verbose: bool=False, warm: Optional[Dict[str, Array]]=None,
              lin_solver: Optional[Any]=None) -> Dict[str, Any]:
        import psutil, os, multiprocessing
        P = _csc(P); q = np.asarray(q, float).ravel()
        n = q.size
        if A is not None:
            A = _csc(A); b = np.asarray(b, float).ravel(); m = b.size
        else:
            m = 0; b = np.zeros(0)

        if warm is None:
            x = np.zeros(n); z = np.zeros(n); u = np.zeros(n); y = np.zeros(m)
        else:
            x = warm.get("x", np.zeros(n)).copy()
            z = warm.get("z", np.zeros(n)).copy()
            u = warm.get("u", np.zeros(n)).copy()
            y = warm.get("y", np.zeros(m)).copy()

        t_total_start = time.perf_counter()
        last_iter_time = 0.0
        t_factor = 0.0; t_x = 0.0; t_z = 0.0; t_dual = 0.0

        t0 = time.perf_counter()
        lin = _ensure_lin_solver(P, A, rho, beta, lin_solver)
        t_factor = time.perf_counter() - t0

        for k in range(1, max_iter+1):
            t_iter0 = time.perf_counter()
            # x-step
            if getattr(lin, "is_kkt", False):
                rhs1 = rho*(z - u) - q
                rhs2 = b - (1.0/beta)*y if m else np.zeros(0)
                tx0 = time.perf_counter()
                x, lam = lin.solve(rhs1, rhs2)
                tx_i = time.perf_counter() - tx0
            else:
                rhs = rho*(z - u) - q
                if m:
                    rhs = rhs + A.T @ (beta*b - y)
                tx0 = time.perf_counter()
                x = lin.solve(rhs)
                tx_i = time.perf_counter() - tx0
            t_x += tx_i

            # z prox
            tmid = alpha*x + (1 - alpha)*z
            v = tmid + u
            tz0 = time.perf_counter()
            z = prox_z(v, 1.0/rho)
            t_z += time.perf_counter() - tz0

            # duals
            td0 = time.perf_counter()
            u = u + (tmid - z)
            if m: y = y + beta*(A @ x - b)
            t_dual += time.perf_counter() - td0

            # residuals
            pri = np.linalg.norm(x - z)
            dua = rho * np.linalg.norm(z - v)
            eqr = np.linalg.norm(A @ x - b) if m else 0.0

            eps_pri = atol*np.sqrt(n) + rtol*max(np.linalg.norm(x), np.linalg.norm(z))
            eps_dua = atol*np.sqrt(n) + rtol*np.linalg.norm(u)
            eq_tol  = (atol*np.sqrt(m) + rtol*np.linalg.norm(b)) if m else 0.0

            last_iter_time = time.perf_counter() - t_iter0
            if verbose and (k == 1 or k % 25 == 0):
                print(f"iter {k:4d}  pri={pri:.2e} dua={dua:.2e} eq={eqr:.2e} ")
            if pri <= eps_pri and dua <= eps_dua and (eqr <= eq_tol):
                break

        t_total = time.perf_counter() - t_total_start
        return {
            "x": x, "z": z, "u": u, "y": y,
            "iters": k, "pri_res": pri, "dual_res": dua, "eq_res": eqr,
            "stats": {
                "total_time": t_total,
                "factor_time": t_factor,
                "last_iter_time": last_iter_time,
                "x_time": t_x, "z_time": t_z, "dual_time": t_dual,
            },
        }


class DouglasRachfordStrategy(BaseStrategy):
    def solve(self, P, q, *, A=None, b=None, prox_z: Prox=None,
              rho: float=1.0, beta: float=0.0, alpha: float=1.0,
              max_iter: int=300, rtol: float=1e-4, atol: float=1e-6,
              verbose: bool=False, warm: Optional[Dict[str, Array]]=None,
              lin_solver: Optional[Any]=None) -> Dict[str, Any]:
        """
        Douglas–Rachford splitting for min f(x) + g(x) with f(x)=0.5 x^T P x + q^T x + I_{Ax=b}(x) and g via prox_z.
        Simplified variant: reflect through prox of f (linear solve) and prox of g.
        """
        P = _csc(P); q = np.asarray(q, float).ravel()
        n = q.size
        if A is not None:
            A = _csc(A); b = np.asarray(b, float).ravel(); m = b.size
        else:
            m = 0; b = np.zeros(0)

        if warm is None:
            x = np.zeros(n); y = np.zeros(n)
        else:
            x = warm.get("x", np.zeros(n)).copy()
            y = warm.get("y", np.zeros(n)).copy()

        lin = _ensure_lin_solver(P, A, rho, beta, lin_solver)

        def prox_f(v, stepsize):
            if getattr(lin, "is_kkt", False):
                rhs1 = stepsize * v - q
                rhs2 = b if m else np.zeros(0)
                u, _ = lin.solve(rhs1, rhs2)
                return u
            rhs = stepsize * v - q
            if m:
                rhs = rhs + A.T @ (beta*b)
            return lin.solve(rhs)

        t_total_start = time.perf_counter()
        last_iter_time = 0.0
        for k in range(1, max_iter+1):
            t0 = time.perf_counter()
            u = prox_f(y, rho)
            w = 2*u - y
            z = prox_z(w, 1.0/rho)
            y = y + alpha * (z - u)
            x = z

            pri = np.linalg.norm(z - u)
            dua = np.linalg.norm(y - w)
            eqr = np.linalg.norm(A @ x - b) if m else 0.0
            eps_pri = atol*np.sqrt(n) + rtol*max(np.linalg.norm(z), np.linalg.norm(u))
            eps_dua = atol*np.sqrt(n) + rtol*np.linalg.norm(y)
            eq_tol  = (atol*np.sqrt(m) + rtol*np.linalg.norm(b)) if m else 0.0
            last_iter_time = time.perf_counter() - t0
            if verbose and (k == 1 or k % 25 == 0):
                print(f"iter {k:4d}  pri={pri:.2e} dua={dua:.2e} eq={eqr:.2e}")
            if pri <= eps_pri and dua <= eps_dua and (eqr <= eq_tol):
                break

        t_total = time.perf_counter() - t_total_start
        return {
            "x": x, "z": x, "u": y, "y": np.zeros(m),
            "iters": k, "pri_res": pri, "dual_res": dua, "eq_res": eqr,
            "stats": {
                "total_time": t_total,
                "last_iter_time": last_iter_time,
            },
        } 