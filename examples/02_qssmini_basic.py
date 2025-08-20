import numpy as np, scipy.sparse as sp
from sepmi.qssmini import solve_qssmini

n = 100
rng = np.random.default_rng(0)
P = sp.eye(n, format="csc")
q = -rng.standard_normal(n)
A = sp.csc_matrix(np.ones((1, n)))
b = np.array([10.0])

gspec = [
    {"g": "abs",      "range": (0, n//2), "args": {"weight": 0.05}},      # L1 on first half
    {"g": "is_pos",   "range": (0, n)},                                   # x >= 0
    {"g": "is_bound", "range": (n//2, n), "args": {"lb": 0.0, "ub": 1.0}},# box on second half
    # {"g": "card",   "range": (0, n), "args": {"weight": 0.01}},         # enable to test L0 heuristic
]

res = solve_qssmini({"P": P, "q": q, "A": A, "b": b, "g": gspec},
                    rho=1.0, beta=5.0, alpha=1.6, verbose=True)

print("iters:", res["iters"], "eq_res:", res["eq_res"], "pri_res:", res["pri_res"])
print("x head:", res["x"][:10])
