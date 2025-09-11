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

def prox_huber(v, t, *, delta=1.0, weight=1.0):
    """
    prox_{t * weight * Huber_delta}(v)
    Huber(z) = 0.5 z^2          if |z| <= delta
               delta*(|z|-0.5*delta) otherwise
    Closed form per component.
    """
    tau = t * weight
    z = v.copy()
    # For |v| <= delta + tau  -> quadratic regime shrinkage
    a = delta + tau
    mask_q = np.abs(v) <= a
    z[mask_q] = v[mask_q] / (1.0 + tau)
    # For |v| > a -> linear regime soft-threshold by tau*delta
    s = np.sign(v[~mask_q])
    z[~mask_q] = v[~mask_q] - s * (tau * delta)
    return z

def prox_group_l2(v, t, *, weight=1.0):
    """
    Block soft-threshold on the WHOLE slice passed in v.
    z = (1 - (t*weight)/||v||_2)_+ v
    """
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return v
    scale = max(0.0, 1.0 - (t*weight)/nrm)
    return scale * v

# --- additional prox/projection utilities ---
def project_simplex(y: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Project y onto {x >= 0, sum x = s}. (Duchi et al., 2008)
    """
    if s <= 0:
        z = np.zeros_like(y)
        return z
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - s
    rho_idx = np.nonzero(u - cssv / (np.arange(1, y.size + 1)) > 0)[0]
    if rho_idx.size == 0:
        z = np.zeros_like(y)
        z[np.argmax(y)] = s
        return z
    rho = rho_idx[-1]
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(y - theta, 0.0)
    return w


def prox_exact_k_simplex(v: np.ndarray, t: float, *, K: int) -> np.ndarray:
    """
    Project v onto Δ_K = { z >= 0, 1^T z = 1, ||z||_0 = K }.
    Deterministic greedy-topK with re-fill if some coordinates collapse to zero.
    """
    n = v.size
    K_eff = max(0, min(int(K), n))
    if K_eff == 0:
        z = np.zeros_like(v)
        return z
    order = np.argsort(v)[::-1]
    S = list(order[:K_eff])
    R = list(order[K_eff:])

    def project_on(S_idx):
        zloc = np.zeros_like(v)
        zS = project_simplex(v[S_idx], s=1.0)
        zloc[S_idx] = zS
        return zloc, int((zS > 0).sum())

    z, nzpos = project_on(S)
    while nzpos < K_eff and len(R) > 0:
        zero_in_S = [i for i in S if z[i] <= 0.0]
        if not zero_in_S:
            break
        take = min(len(zero_in_S), len(R))
        for j in range(take):
            idx_zero = zero_in_S[j]
            pos = S.index(idx_zero)
            S[pos] = R[j]
        R = R[take:]
        z, nzpos = project_on(S)
        if take == 0:
            break
    return z


def prox_s_band(vs: float, t: float, *, C: float, R: float) -> float:
    """Projection of scalar s onto [C-R, C]."""
    lo, hi = C - R, C
    if vs < lo: return lo
    if vs > hi: return hi
    return float(vs)


def prox_d_grid_min_dollars(
    vd: np.ndarray, t: float, *, prices: np.ndarray, min_dollars: float, decimals: int
) -> np.ndarray:
    """
    Elementwise projection onto S_i = {0} ∪ {k g_i : k ∈ N, k g_i ≥ min_dollars},
    where g_i = p_i * q and q = 10^{-decimals}.
    """
    p = np.asarray(prices, float)
    q = 10.0 ** (-max(0, int(decimals)))
    g = p * q
    v = np.asarray(vd, float)
    out = np.empty_like(v)

    good = (g > 0) & np.isfinite(g) & np.isfinite(v)
    out[~good] = 0.0

    idx = np.where(good)[0]
    for i in idx:
        gi = g[i]; vi = v[i]
        k = np.round(vi / gi)
        z = k * gi  # nearest signed multiple
        az = abs(z)
        if az == 0 or az >= min_dollars:
            out[i] = z
            continue
        kmin = np.ceil(min_dollars / gi)
        zmin = kmin * gi
        cand0 = 0.0
        candm = np.sign(z) * zmin
        out[i] = candm if abs(vi - candm) < abs(vi - cand0) else cand0
    
    return out

def prox_u_integer_min_dollars(
    vu: np.ndarray, t: float, *, prices: np.ndarray, min_dollars: float
) -> np.ndarray:
    """
    Elementwise projection onto U_i = {0} ∪ {k ∈ Z : p_i |k| ≥ min_dollars}.
    """
    p = np.asarray(prices, float)
    v = np.asarray(vu, float)
    out = np.empty_like(v)

    good = (p > 0) & np.isfinite(p) & np.isfinite(v)
    out[~good] = 0.0

    idx = np.where(good)[0]
    for i in idx:
        pi = p[i]; vi = v[i]
        k = np.round(vi)
        if k == 0 or (pi * abs(k) >= min_dollars):
            out[i] = k
            continue
        kmin = np.ceil(min_dollars / pi)
        cand0 = 0.0
        candm = np.sign(vi) * kmin
        out[i] = candm if abs(vi - candm) < abs(vi - cand0) else cand0

    return out

# registry (name -> function)
PROX_REGISTRY = {
    "abs": prox_abs,              # L1
    "is_pos": prox_is_pos,        # x >= 0
    "is_bound": prox_is_bound,    # lb <= x <= ub
    "is_zero": prox_is_zero,      # x = 0
    "card": prox_card,            # L0 (heuristic)
    "huber": prox_huber,          # Huber
    "group_l2": prox_group_l2,    # Group L2
    # extra proxes
    "exact_k_simplex": prox_exact_k_simplex,
    "s_band": prox_s_band,
    "d_grid_min_dollars": prox_d_grid_min_dollars,
    "u_integer_min_dollars": prox_u_integer_min_dollars,
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
