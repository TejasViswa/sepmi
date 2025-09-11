# sepmi/cvxpy_bridge.py
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

from .prox_catalog import make_dispatch_from_gspec
from .qssmini import solve_qssmini


def solve_qp_into_cvxpy(
    var, 
    P,
    q,
    *,
    A=None,
    b=None,
    gspec: Optional[List[Dict[str, Any]]] = None,
    strategy: str = "admm",
    solver: str = "auto",
    **solver_kwargs,
) -> Dict[str, Any]:
    """
    Solve a QP with SEPMI and assign the result into a cvxpy.Variable.

    Parameters
    - var: cvxpy.Variable (1-D)
    - P, q: quadratic and linear term (numpy/scipy)
    - A, b: optional equality constraints A x = b
    - gspec: list of prox specs for separable g, per make_dispatch_from_gspec
    - strategy: currently only "admm" is supported (future: DR)
    - solver: linear system backend mode passed to qssmini
    - **solver_kwargs: forwarded to solve_qssmini (e.g., rho, beta, alpha, max_iter, rtol, atol)

    Returns
    - res: the SEPMI result dict; also sets var.value = res["x"].
    """
    if gspec is None:
        gspec = []
    data = {"P": P, "q": q, "A": A, "b": b, "g": gspec}
    if strategy != "admm":
        # For now, route through qssmini (ADMM). Strategy routing can be added later if needed.
        pass
    res = solve_qssmini(
        data,
        solver=solver,
        **solver_kwargs,
    )
    x = np.asarray(res.get("x", res.get("z")), float)
    try:
        var.value = x
    except Exception:
        # If var is not a cvxpy.Variable, ignore assignment
        pass
    return res


def assign_solution(var, res: Dict[str, Any], key: str = "x") -> None:
    """
    Assign a solution vector from a SEPMI result dict into a cvxpy.Variable.
    """
    x = np.asarray(res.get(key), float)
    try:
        var.value = x
    except Exception:
        pass 