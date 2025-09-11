# sepmi/polish.py
import numpy as np


def polish_to_band(
    d: np.ndarray, prices: np.ndarray, t: np.ndarray,
    C: float, R: float, min_dollars: float, decimals: int,
    max_steps: int = 100000,
) -> np.ndarray:
    """
    Greedy discrete projection of spend vector d onto the budget band [C-R, C],
    respecting per-name set S_i = {0} ∪ {k g_i: k ∈ N, k g_i ≥ min_dollars},
    where g_i = p_i * q and q = 10^{-decimals}. Chooses at each step the move
    with smallest increase in |d - t| per dollar changed.
    """
    d = np.array(d, float, copy=True)
    p = np.asarray(prices, float)
    q = 10.0 ** (-max(0, int(decimals)))
    g = p * q
    good = (g > 0) & np.isfinite(g) & np.isfinite(d) & np.isfinite(t)

    def _total():
        return float(np.sum(d))

    def _one_down(i):
        di, gi = d[i], g[i]
        if di <= 0 or gi <= 0:
            return di, 0.0
        if di - gi >= max(0.0, min_dollars):
            d_next = di - gi
        else:
            d_next = 0.0
        step = di - d_next
        return (d_next, step) if step > 0 else (di, 0.0)

    def _one_up(i):
        di, gi = d[i], g[i]
        if gi <= 0:
            return di, 0.0
        if di <= 0:
            kmin = np.ceil(min_dollars / gi) if min_dollars > 0 else 1.0
            d_next = float(max(gi, kmin * gi))
        else:
            d_next = di + gi
        step = d_next - di
        return (d_next, step) if step > 0 else (di, 0.0)

    steps = 0
    while _total() > C + 1e-9 and steps < max_steps:
        idx = np.where(good & (d > 0))[0]
        if idx.size == 0:
            break
        d_next = np.empty_like(d[idx])
        step   = np.empty_like(d[idx])
        for k,j in enumerate(idx):
            dj, sj = _one_down(j)
            d_next[k], step[k] = dj, sj
        valid = step > 0
        if not np.any(valid):
            break
        idx, d_next, step = idx[valid], d_next[valid], step[valid]
        cur_dev = np.abs(d[idx] - t[idx])
        new_dev = np.abs(d_next - t[idx])
        cost_per_dollar = (new_dev - cur_dev) / step
        j = idx[np.argmin(cost_per_dollar)]
        dj, sj = _one_down(j)
        d[j] = dj
        steps += 1

    steps_up = 0
    target_min = C - R
    while _total() < target_min - 1e-9 and steps + steps_up < max_steps:
        idx = np.where(good)[0]
        if idx.size == 0:
            break
        cands = []
        total_now = _total()
        for j in idx:
            dj, sj = _one_up(j)
            if sj <= 0:
                continue
            if total_now + sj <= C + 1e-9:
                cands.append((j, dj, sj))
        if not cands:
            break
        ratios = []
        for (j, dj, sj) in cands:
            cur = abs(d[j] - t[j]); new = abs(dj - t[j])
            ratios.append((new - cur) / sj)
        j, dj, sj = cands[int(np.argmin(ratios))]
        d[j] = dj
        steps_up += 1

    return d 