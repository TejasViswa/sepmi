# sepmi/prox_catalog.py
import numpy as np

# --- base proxes (separable, vectorized) ---
def prox_abs(v, t, *, weight=1.0):
    """prox_{t * weight * |.|}(v)  -> soft-threshold"""
    tau = t * weight
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def prox_is_pos(v, t, **_):
    """indicator of x >= 0"""
    return np.maximum(v, 0.0)

def prox_is_bound(v, t, *, lb=None, ub=None, **_):
    """indicator of lb <= x <= ub"""
    lo = -np.inf if lb is None else lb
    hi =  np.inf if ub is None else ub
    return np.clip(v, lo, hi)

def prox_is_zero(v, t, **_):
    """indicator of x = 0"""
    z = np.zeros_like(v)
    return z

def prox_card(v, t, *, weight=1.0):
    """prox of weight * ||x||_0 (hard-threshold) — nonconvex heuristic."""
    # scalar (per-entry) hard threshold: |v| <= sqrt(2*t*weight) -> 0
    thr = np.sqrt(2.0 * t * weight)
    z = v.copy()
    z[np.abs(v) <= thr] = 0.0
    return z

# registry (name -> function)
PROX_REGISTRY = {
    "abs": prox_abs,              # L1
    "is_pos": prox_is_pos,        # x >= 0
    "is_bound": prox_is_bound,    # lb <= x <= ub
    "is_zero": prox_is_zero,      # x = 0
    "card": prox_card,            # L0 (heuristic)
}

def make_dispatch_from_gspec(n, gspec):
    """
    Build a prox dispatcher f(v, t) for a length-n vector, from a list of specs.
    Each spec: {"g": <name>, "range": (start, end), "args": {...}}
    Unspecified indices use identity (prox of 0).
    """
    # default: identity prox
    segments = []
    covered = np.zeros(n, dtype=bool)

    for item in gspec:
        name = item["g"]
        (start, end) = item["range"]  # Python slice [start:end]
        args = item.get("args", {})
        prox_fn = PROX_REGISTRY[name]

        mask = np.zeros(n, dtype=bool)
        mask[start:end] = True
        covered |= mask

        # capture a view mask + args
        segments.append((mask, prox_fn, args))

    # any index not covered -> identity prox
    if not covered.all():
        mask = ~covered
        segments.append((mask, lambda v, t, **_: v, {}))  # identity prox

    def prox_dispatch(v, t):
        z = v.copy()
        for mask, fn, args in segments:
            if mask.any():
                z[mask] = fn(v[mask], t, **args)
        return z

    return prox_dispatch
