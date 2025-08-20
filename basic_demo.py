import numpy as np, scipy.sparse as sp
from sepmi import solve
n = 300
rng = np.random.default_rng(0)
P = sp.eye(n, format="csc")
q = -rng.standard_normal(n)
A = sp.csc_matrix(np.ones((1, n)))
b = np.array([15.0])
sol = solve(P, q, A=A, b=b, lam_l1=0.05, K_card=25, rho=1.0, beta=5.0,
            alpha=1.6, max_iter=300, verbose=True)
print("Objective:", sol["obj"])
print("Eq residual:", sol["eq_res"])
print("Consensus residual:", sol["pri_res"])
print("Nonzeros in z:", (np.abs(sol["z"]) > 0).sum())
