# sepmi.py — minimal separable ADMM solver (Python-only MVP)
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

def _ensure_csc(M):
    if M is None: return None
    return M if sp.isspmatrix_csc(M) else M.tocsc()

class _NormalEqSolver:
    def __init__(self, P, A=None, rho=1.0, beta=1.0):
        P = _ensure_csc(P); n = P.shape[0]
        I = sp.identity(n, format="csc")
        AtA = (A.T @ A).tocsc() if A is not None else sp.csc_matrix((n,n))
        K = (P + rho*I + beta*AtA).tocsc()
        self._solve = spla.factorized(K)
    def solve(self, rhs): return self._solve(rhs)

def prox_l1(v, t): return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)
def prox_box(v, lo=None, hi=None):
    if lo is None and hi is None: return v
    if lo is None: return np.minimum(v, hi)
    if hi is None: return np.maximum(v, lo)
    return np.clip(v, lo, hi)
def prox_l1_box(v, lam_over_rho, lo=None, hi=None):
    return prox_box(prox_l1(v, lam_over_rho), lo, hi)
def prox_l0_l1_topk(v, K, lam_over_rho):
    s = prox_l1(v, lam_over_rho)
    cost_on  = 0.5*(s - v)**2 + lam_over_rho*np.abs(s)
    cost_off = 0.5*(v**2)
    benefit  = cost_off - cost_on
    z = np.zeros_like(v)
    pos = np.flatnonzero(benefit > 0)
    if pos.size == 0: return z
    if K >= v.size or pos.size <= K:
        z[pos] = s[pos]; return z
    idx = np.argpartition(benefit[pos], -K)[-K:]
    keep = pos[idx]; z[keep] = s[keep]; return z

def solve(P, q, A=None, b=None, *, lam_l1=0.0, box=None, K_card=None,
          rho=1.0, beta=5.0, alpha=1.6, max_iter=300, rtol=1e-4, atol=1e-6,
          verbose=False, warm_start=None):
    P = _ensure_csc(P); q = np.asarray(q, float).ravel(); n = q.size
    if A is not None:
        A = _ensure_csc(A); b = np.asarray(b, float).ravel(); m = b.size
    else:
        m = 0; b = np.zeros(0)
    lo, hi = (box if box is not None else (None, None))
    if warm_start is None:
        x = np.zeros(n); z = np.zeros(n); u = np.zeros(n); y = np.zeros(m)
    else:
        x = warm_start.get("x", np.zeros(n)).copy()
        z = warm_start.get("z", np.zeros(n)).copy()
        u = warm_start.get("u", np.zeros(n)).copy()
        y = warm_start.get("y", np.zeros(m)).copy()
    # lin = _NormalEqSolver(P, A, rho=rho, beta=beta)
    if A is not None and A.shape[0] <= 2000:   # tweak threshold as you like
        lin = _WoodburySolver(P, A, rho=rho, beta=beta)
    else:
        lin = _NormalEqSolver(P, A, rho=rho, beta=beta)
    def obj(x_, z_): return 0.5 * x_ @ (P @ x_) + q @ x_ + lam_l1*np.sum(np.abs(z_))
    for k in range(1, max_iter+1):
        rhs = rho*(z - u) - q
        if m: rhs = rhs + A.T @ (beta*b - y)
        x = lin.solve(rhs)
        t = alpha*x + (1 - alpha)*z; v = t + u
        lam_over_rho = (lam_l1 / rho) if rho > 0 else lam_l1
        if K_card is None:
            z = prox_l1_box(v, lam_over_rho, lo, hi)
        else:
            z = prox_l0_l1_topk(v, K_card, lam_over_rho)
            if box is not None:
                z = np.clip(z, lo if lo is not None else -np.inf,
                               hi if hi is not None else  np.inf)
        u = u + (t - z)
        if m: y = y + beta*(A @ x - b)
        pri = np.linalg.norm(x - z)
        dua = rho * np.linalg.norm(z - v)
        eqr = np.linalg.norm(A @ x - b) if m else 0.0
        eps_pri = atol*np.sqrt(n) + rtol*max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dua = atol*np.sqrt(n) + rtol*np.linalg.norm(u)
        eq_tol  = (atol*np.sqrt(m) + rtol*np.linalg.norm(b)) if m else 0.0
        if verbose and (k == 1 or k % 25 == 0):
            print(f"iter {k:4d}  obj={obj(x,z):.6f}  pri={pri:.2e}  dua={dua:.2e}  eq={eqr:.2e}")
        if pri <= eps_pri and dua <= eps_dua and (eqr <= eq_tol): break
    return {"x":x,"z":z,"u":u,"y":y,"obj":obj(x,z),"pri_res":pri,"dual_res":dua,"eq_res":eqr,"iters":k}

class _WoodburySolver:
    """
    (H + beta A^T A)^{-1} = H^{-1} - H^{-1} A^T (I + beta A H^{-1} A^T)^{-1} beta A H^{-1}
    with H = P + rho I. Use when m = rows(A) is small/moderate.
    """
    def __init__(self, P, A, rho=1.0, beta=1.0):
        P = _ensure_csc(P); A = _ensure_csc(A)
        n = P.shape[0]
        I = sp.identity(n, format="csc")
        H = (P + rho * I).tocsc()
        self._Hsolve = spla.factorized(H)
        self.A = A
        self.beta = float(beta)

        AT = A.T.toarray()          # (n, m)
        self.X = self._Hsolve(AT)   # (n, m), H X = A^T

        S = np.eye(A.shape[0]) + self.beta * (A @ self.X)  # (m, m)
        self._S_cho = la.cho_factor(S, lower=True, check_finite=False)

    def solve(self, rhs):
        y = self._Hsolve(rhs)                              # H^{-1} rhs
        t = self.beta * (self.A @ y)                       # (m,)
        s = la.cho_solve(self._S_cho, t, check_finite=False)
        return y - self.X @ s


