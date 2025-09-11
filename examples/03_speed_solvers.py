# Run with:  python -m sepmi.examples.03_speed_solvers
import os, time
import numpy as np
import scipy.sparse as sp
from sepmi.qssmini import solve_qssmini

# (Optional) pin threads early
ncores = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(ncores))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ncores))
os.environ.setdefault("MKL_NUM_THREADS", str(ncores))
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

rng = np.random.default_rng(0)

def make_problem(n, m, lam=0.05, density=0.01, hard=False, seed=0):
    """
    QP + equality:
        min 0.5 * x^T P x + q^T x + lam * ||x||_1   s.t. A x = b
    When hard=True, make the instance non-trivial:
      - random q (so x*=0 is no longer optimal)
      - random b (so Ax=b is not trivially satisfied)
      - P = I + G^T G (SPD, not diagonal)
      - A denser by default
    """
    rng = np.random.default_rng(seed)

    if hard:
        # P = I + G^T G (mildly ill-conditioned but SPD), ~1–5% column density
        k = max(1, int(0.05 * n))
        G = sp.random(k, n, density=0.02, data_rvs=rng.standard_normal, format="csc")
        P = (sp.identity(n, format="csc") + G.T @ G).tocsc()
        q = rng.standard_normal(n)
        # Random sparse A with denser pattern
        d = max(density, 0.005)   # ensure it’s not tiny
        A = sp.random(m, n, density=d, data_rvs=rng.standard_normal, format="csc") if m else None
        if m:
            # optional row scaling
            row_norms = np.sqrt((A.multiply(A)).sum(axis=1)).A.ravel() + 1e-12
            A = sp.diags(1.0 / row_norms) @ A
        b = rng.standard_normal(m) if m else np.zeros(0)
    else:
        P = sp.identity(n, format="csc")
        q = np.zeros(n)
        A = sp.random(m, n, density=density, data_rvs=rng.standard_normal, format="csc") if m else None
        if m:
            row_norms = np.sqrt((A.multiply(A)).sum(axis=1)).A.ravel() + 1e-12
            A = sp.diags(1.0 / row_norms) @ A
        b = np.zeros(m)

    gspec = [ {"g": "abs", "range": (0, n), "args": {"weight": lam}} ]
    return {"P": P, "q": q, "A": A, "b": b, "g": gspec}


def run_case(tag, data, solver, iters=150, rho=1.0, beta=2.0, alpha=1.6):
    t0 = time.perf_counter()
    res = solve_qssmini(
        data,
        rho=rho, beta=beta, alpha=alpha, max_iter=iters,
        verbose=False,                 # no per-iter residual prints
        profile=True,                  # record times internally
        profile_interval=25,           # <- no [stats] lines
        print_header=False,            # <- no header
        solver=solver
    )
    t1 = time.perf_counter()
    print(f"[{tag:20s}] solver={solver:8s}  iters={res['iters']:4d}  "
          f"time={t1-t0:6.3f}s  factor={res['stats'].get('factor_time',0.0):6.3f}s  "
          f"pri={res['pri_res']:.2e} eq={res['eq_res']:.2e}")
    return res, t1 - t0


def main():
    # ---------- Case A: few equalities (Woodbury should shine) ----------
    nA, mA, densA = 20000, 400, 0.01
    data_small = make_problem(nA, mA, lam=0.05, density=densA)
    print("\n=== Case A: few equalities (m << n) ===")
    run_case("few-eq / woodbury", data_small, solver="woodbury")
    run_case("few-eq / kkt_petsc", data_small, solver="kkt_petsc")

    # ---------- Case B: many equalities (tractable & fair) ----------
    nB, mB, densB = 15000, 3000, 0.001
    data_large = make_problem(nB, mB, lam=0.05, density=densB)
    print("\n=== Case B: many equalities (skip Woodbury) ===")
    run_case("many-eq / normal_cholmod", data_large, solver="normal_cholmod")
    run_case("many-eq / kkt_petsc", data_large, solver="kkt_petsc")
    
    # ---------- Case C: n=5000, no equalities (SPD) ----------
    nC = 5000
    data_spd = make_problem(nC, 0, lam=0.05, density=0.0, hard=False)
    print("\n=== Case C: n=5000, no equalities (SPD) ===")
    run_case("spd / normal_cholmod", data_spd, solver="normal_cholmod")
    run_case("spd / kkt_petsc     ", data_spd, solver="kkt_petsc")  # PETSc handles m=0

    # ---------- Case C(hard): n=5000, no equalities, harder P & q ----------
    data_spd_h = make_problem(nC, 0, lam=0.05, density=0.0, hard=True)
    print("\n=== Case C(hard): n=5000, no eq, random q, P=I+G^T G ===")
    run_case("spd-hard / normal_cholmod", data_spd_h, solver="normal_cholmod", iters=150, rho=1.0, beta=2.0, alpha=1.6)
    run_case("spd-hard / kkt_petsc    ", data_spd_h, solver="kkt_petsc",      iters=150, rho=1.0, beta=2.0, alpha=1.6)

    # ---------- Case C2(hard): n=5000, m=5000 (indefinite KKT) ----------
    nC2, mC2, densC2 = 5000, 5000, 0.01  # ~250k nnz in A
    data_many_h = make_problem(nC2, mC2, lam=0.05, density=densC2, hard=True)
    print("\n=== Case C2(hard): n=5000, m=5000, denser A, random q,b ===")
    run_case("many-eq-hard / kkt_petsc   ", data_many_h, solver="kkt_petsc",      iters=150, rho=1.0, beta=2.0, alpha=1.6)
    run_case("many-eq-hard / normal_chol ", data_many_h, solver="normal_cholmod", iters=150, rho=1.0, beta=2.0, alpha=1.6)

if __name__ == "__main__":
    main()
