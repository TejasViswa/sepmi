"""SEPMI - Separable ADMM solver package"""
from .qssmini import solve_qssmini
from .strategies import ADMMStrategy, DouglasRachfordStrategy
from .linear_solvers import (
    NormalEqSolver, CholmodNormalSolver, WoodburySolver,
    KKTSolver, PetscKKTSolver, CholmodNormalSolverReusable
)
from .cvxpy_bridge import solve_qp_into_cvxpy, assign_solution

__version__ = "0.1.0"
__all__ = [
    "solve_qssmini",
    "ADMMStrategy", "DouglasRachfordStrategy",
    "NormalEqSolver", "CholmodNormalSolver", "WoodburySolver",
    "KKTSolver", "PetscKKTSolver", "CholmodNormalSolverReusable",
    "solve_qp_into_cvxpy", "assign_solution",
]
