# sepmi/qssmini.py
import numpy as np
import scipy.sparse as sp
from .prox_catalog import make_dispatch_from_gspec
from .strategies import ADMMStrategy
from .linear_solvers import (
    NormalEqSolver, CholmodNormalSolver, WoodburySolver,
    KKTSolver, PetscKKTSolver, CholmodNormalSolverReusable
)


def _choose_lin_solver(P, A, rho, beta, mode):
    if mode == "normal":
        return NormalEqSolver(P, A, rho=rho, beta=beta)
    if mode == "normal_cholmod":
        return CholmodNormalSolver(P, A, rho=rho, beta=beta,
                                   cholmod_mode="supernodal",
                                   ordering_method="best")
    if mode == "woodbury":
        if A is None or A.shape[0] == 0:
            return NormalEqSolver(P, A, rho=rho, beta=beta)
        return WoodburySolver(P, A, rho=rho, beta=beta)
    if mode == "kkt":
        if A is None or A.shape[0] == 0:
            return NormalEqSolver(P, A, rho=rho, beta=beta)
        return KKTSolver(P, A, rho=rho, beta=beta)
    if mode == "kkt_petsc":
        return PetscKKTSolver(P, A, rho=rho, beta=beta)
    if mode == "kkt_mumps":
        # kept in admm_core before; if needed can be added here by import
        from .linear_solvers import MumpsKKTSolver
        return MumpsKKTSolver(P, A, rho=rho, beta=beta)

    # auto policy
    if A is None or A.shape[0] == 0:
        try:
            return CholmodNormalSolver(P, A, rho=rho, beta=beta,
                                       cholmod_mode="supernodal",
                                       ordering_method="best")
        except Exception:
            return NormalEqSolver(P, A, rho=rho, beta=beta)
    m, n = A.shape[0], P.shape[0]
    if m and (m <= 2000 and m <= n // 4):
        return WoodburySolver(P, A, rho=rho, beta=beta)
    try:
        return PetscKKTSolver(P, A, rho=rho, beta=beta)
    except Exception:
        try:
            return CholmodNormalSolver(P, A, rho=rho, beta=beta,
                                       cholmod_mode="supernodal",
                                       ordering_method="best")
        except Exception:
            return KKTSolver(P, A, rho=rho, beta=beta)


def solve_qssmini(data, *, rho=1.0, beta=5.0, alpha=1.6, max_iter=300,
                  profile=True, profile_interval=25, print_header=True,
                  rtol=1e-4, atol=1e-6, verbose=False, solver="auto"):
    """
    data = {
      "P": <n x n sparse SPD-ish>,
      "q": <n>,
      "A": <m x n sparse> (optional),
      "b": <m> (optional),
      "g": [ {"g": "<name>", "range": (i0, i1), "args": {...}}, ...]
    }
    """
    P = data["P"]; q = np.asarray(data["q"], float).ravel()
    A = data.get("A"); b = data.get("b")
    gspec = data.get("g", [])
    n = q.size

    prox = make_dispatch_from_gspec(n, gspec)

    lin = _choose_lin_solver(P, A, rho, beta, solver)
    strat = ADMMStrategy()
    return strat.solve(P, q, A=A, b=b, prox_z=prox,
                       rho=rho, beta=beta, alpha=alpha,
                       max_iter=max_iter, rtol=rtol, atol=atol,
                       verbose=verbose, warm=None, lin_solver=lin)
