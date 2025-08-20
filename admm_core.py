# sepmi/admm_core.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _csc(M):
    return M if sp.isspmatrix_csc(M) else M.tocsc()

class NormalEqSolver:
    """MVP: factor K = P + rho I + beta A^T A once; later we’ll add Woodbury/KKT."""
    def __init__(self, P, A=None, rho=1.0, beta=1.0):
        P = _csc(P); n = P.shape[0]
        I = sp.identity(n, format="csc")
        AtA = (_csc(A).T @ _csc(A)) if A is not None else sp.csc_matrix((n, n))
        K = (P + rho*I + beta*AtA).tocsc()
        self._solve = spla.factorized(K)
        self.A = A
        self.beta = float(beta)

    def solve(self, rhs):
        return self._solve(rhs)

def admm_solve(P, q, *, A=None, b=None, prox_z=None,
               rho=1.0, beta=5.0, alpha=1.6,
               max_iter=300, rtol=1e-4, atol=1e-6, verbose=False, warm=None):
    """
    Min 0.5 x^T P x + q^T x + sum_i g_i(z_i)
    s.t. A x = b,  x = z

    prox_z(v, t) must return prox_{t * g}(v).
    """
    P = _csc(P); q = np.asarray(q, float).ravel()
    n = q.size
    if A is not None:
        A = _csc(A); b = np.asarray(b, float).ravel(); m = b.size
    else:
        m = 0; b = np.zeros(0)

    # state
    if warm is None:
        x = np.zeros(n); z = np.zeros(n); u = np.zeros(n); y = np.zeros(m)
    else:
        x = warm.get("x", np.zeros(n)).copy()
        z = warm.get("z", np.zeros(n)).copy()
        u = warm.get("u", np.zeros(n)).copy()
        y = warm.get("y", np.zeros(m)).copy()

    lin = NormalEqSolver(P, A, rho=rho, beta=beta)

    def objective(x_, z_):
        return 0.5 * x_ @ (P @ x_) + q @ x_  # (we don’t try to evaluate sum g_i here)

    for k in range(1, max_iter+1):
        # x-step: (P + rho I + beta A^T A) x = rho(z - u) - q + A^T (beta b - y)
        rhs = rho*(z - u) - q
        if m: rhs = rhs + A.T @ (beta*b - y)
        x = lin.solve(rhs)

        # z-step: separable prox on v = (alpha x + (1-alpha) z) + u
        t = alpha*x + (1 - alpha)*z
        v = t + u
        z = prox_z(v, rho)  # note: t = rho, scale handled inside prox_z

        # duals
        u = u + (t - z)
        if m: y = y + beta*(A @ x - b)

        # residuals
        pri = np.linalg.norm(x - z)
        dua = rho * np.linalg.norm(z - v)
        eqr = np.linalg.norm(A @ x - b) if m else 0.0

        eps_pri = atol*np.sqrt(n) + rtol*max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dua = atol*np.sqrt(n) + rtol*np.linalg.norm(u)
        eq_tol  = (atol*np.sqrt(m) + rtol*np.linalg.norm(b)) if m else 0.0

        if verbose and (k == 1 or k % 25 == 0):
            print(f"iter {k:4d}  pri={pri:.2e} dua={dua:.2e} eq={eqr:.2e}")

        if pri <= eps_pri and dua <= eps_dua and (eqr <= eq_tol):
            break

    return {"x": x, "z": z, "u": u, "y": y, "iters": k,
            "pri_res": pri, "dual_res": dua, "eq_res": eqr}
