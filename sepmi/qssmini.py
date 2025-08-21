# sepmi/qssmini.py
import numpy as np
import scipy.sparse as sp
from .prox_catalog import make_dispatch_from_gspec
from .admm_core import admm_solve

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

    return admm_solve(P, q, A=A, b=b, prox_z=prox,
                      rho=rho, beta=beta, alpha=alpha,
                      max_iter=max_iter, rtol=rtol, atol=atol,
                      profile=profile, profile_interval=profile_interval,
                      print_header=print_header,
                      verbose=verbose, solver=solver)
