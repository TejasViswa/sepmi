# sepmi/admm_core.py
from __future__ import annotations
import os, multiprocessing
n = os.cpu_count() or multiprocessing.cpu_count() or 1

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, str(n))     # or os.environ[k] = str(n) to force

# (optional, steadier behavior)
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

import time, os, math, multiprocessing
try:
    import psutil
except Exception:
    psutil = None
try:
    from threadpoolctl import threadpool_info
except Exception:
    threadpool_info = None
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
    from sksparse.cholmod import analyze
except Exception:
    cholmod_cholesky = None
try:
    import mumps as _mumps_mod
    DMumpsContext = getattr(_mumps_mod, "DMumpsContext", None)
    _HAVE_MUMPS = DMumpsContext is not None
except Exception:
    DMumpsContext = None
    _HAVE_MUMPS = False

try:
    from mpi4py import MPI
    _MUMPS_COMM = MPI.COMM_SELF   # single-process; OpenMP threads do the work
except Exception:
    _MUMPS_COMM = None

try:
    from petsc4py import PETSc
    _HAVE_PETSC = True
except Exception:
    _HAVE_PETSC = False
    
def _bytes_h(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _tp_summary():
    rows = []
    if threadpool_info is not None:
        try:
            for info in threadpool_info():
                rows.append({
                    "api": info.get("internal_api"),
                    "prefix": info.get("prefix"),
                    "n_threads": info.get("num_threads"),
                    "lib": os.path.basename(info.get("filepath",""))})
        except Exception:
            pass
    return rows

def _print_thread_env():
    keys = ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"]
    env = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
    return ", ".join(f"{k}={v}" for k,v in env.items()) or "(unset)"

def _csc(M):
    return M if sp.isspmatrix_csc(M) else M.tocsc()

class NormalEqSolver:
    def __init__(self, P, A=None, rho=1.0, beta=1.0):
        P = _csc(P).astype(np.float64); n = P.shape[0]
        I = sp.identity(n, format="csc", dtype=np.float64)
        AtA = sp.csc_matrix((n, n), dtype=np.float64)
        if A is not None:
            A = _csc(A).astype(np.float64)
            AtA = A.T @ A
        K = (P + rho*I + beta*AtA).tocsc()
        self._solve = spla.factorized(K)

    def solve(self, rhs):
        return self._solve(rhs)
class CholmodNormalSolver:
    """
    Multithreaded SPD solve via CHOLMOD on K = P + rho I + beta A^T A.
    """
    def __init__(self, P, A=None, rho=1.0, beta=1.0,
                 cholmod_mode="supernodal",
                 ordering_method="best"):
        if cholmod_cholesky is None:
            raise RuntimeError("CHOLMOD unavailable (install scikit-sparse).")
        P = _csc(P).astype(np.float64); n = P.shape[0]
        I = sp.identity(n, format="csc", dtype=np.float64)
        K = (P + rho*I).tocsc()
        if A is not None and A.shape[0] > 0 and beta != 0.0:
            A = _csc(A).astype(np.float64)
            K = (K + beta * (A.T @ A)).tocsc()
        # ↓ force supernodal + best (or pass-through from caller)
        self._fact = cholmod_cholesky(
            K, mode=cholmod_mode, ordering_method=ordering_method
        )

    def solve(self, rhs):
        return self._fact(rhs)
    
class CholmodNormalSolverReusable:
    def __init__(self, P, rho=1.0, cholmod_mode="supernodal", ordering_method="best"):
        P = _csc(P).astype(np.float64)
        n = P.shape[0]
        I = sp.identity(n, format="csc", dtype=np.float64)
        self.P = P; self.I = I; self.rho = float(rho)
        self._an = analyze(P + 0*I, mode=cholmod_mode, ordering_method=ordering_method)  # symbolic only
        self._fact = self._an.cholesky(P + self.rho*I)  # numeric factor once

    def solve(self, rhs):
        return self._fact(rhs)
class WoodburySolver:
    """
    (H + beta A^T A)^{-1} with H=P+rho I via SMW. Use only when m << n.
    """
    def __init__(self, P, A, rho=1.0, beta=1.0):
        P = _csc(P).astype(np.float64); A = _csc(A).astype(np.float64)
        m = A.shape[0]; n = P.shape[0]
        if m > 2000:
            raise ValueError(f"WoodburySolver: m={m} too large; use KKT/normal.")
        I = sp.identity(n, format="csc", dtype=np.float64)
        H = (P + rho * I).tocsc()
        self._Hsolve = spla.factorized(H)
        self.A = A
        self.beta = float(beta)

        # X = H^{-1} A^T (n x m), S = I + beta * A X (m x m) SPD dense
        AT = A.T.astype(np.float64, copy=False).toarray(order='F')  # dense (n, m)
        self.X = self._Hsolve(AT)                                   # (n, m)
        S = np.eye(m, dtype=np.float64) + self.beta * (A @ self.X)  # (m, m) SPD
        self._S_cho = la.cho_factor(S, lower=True, check_finite=False)

    def solve(self, rhs):
        y = self._Hsolve(rhs)                       # H^{-1} rhs
        t = self.beta * (self.A @ y)                # (m,)
        s = la.cho_solve(self._S_cho, t, check_finite=False)
        return y - self.X @ s

class KKTSolver:
    is_kkt = True
    def __init__(self, P, A, rho=1.0, beta=5.0):
        P = _csc(P).astype(np.float64); A = _csc(A).astype(np.float64)
        n = P.shape[0]; m = A.shape[0]
        I_n = sp.identity(n, format="csc", dtype=np.float64)
        I_m = sp.identity(m, format="csc", dtype=np.float64)
        KKT = sp.bmat([[P + rho*I_n,      A.T],
                       [A,               -(1.0/beta)*I_m]], format="csc")
        # Sparse LU with good ordering (single-threaded SuperLU, but faster/robuster than factorized())
        self._lu = spla.splu(KKT, permc_spec="COLAMD", diag_pivot_thresh=0.0)
        self.n, self.m = n, m

    def solve(self, rhs1, rhs2):
        rhs = np.concatenate([rhs1, rhs2])
        sol = self._lu.solve(rhs)
        return sol[:self.n], sol[self.n:]

class MumpsKKTSolver:
    """
    Symmetric-indefinite KKT factorization with MUMPS (LDL^T, sym=2).
    Provide ONE triangle (upper) with 1-based indices.
    Handles both pymumps APIs:
      1) rows/cols -> analyze, values -> factor -> solve(job=3)
      2) assembled(irn,jcn,a[,N]) -> analyze/factor or run(job=1/2)
    """
    is_kkt = True
    def __init__(self, P, A, rho=1.0, beta=5.0):
        if not _HAVE_MUMPS or DMumpsContext is None:
            raise RuntimeError("pymumps not available (conda-forge: pymumps mumps scotch metis mpi4py).")

        import numpy as np, scipy.sparse as sp
        P = _csc(P).astype(np.float64)
        A = _csc(A).astype(np.float64)
        n = P.shape[0]; m = A.shape[0]; N = n + m

        I_n = sp.identity(n, format="csc", dtype=np.float64)
        I_m = sp.identity(m, format="csc", dtype=np.float64)

        # Blocks → COO
        K11 = (P + rho * I_n).tocoo()
        K12 = A.T.tocoo()                  # (n x m)
        K21 = A.tocoo()                    # (m x n)
        K22 = (-(1.0 / beta) * I_m).tocoo()

        rows = np.concatenate([K11.row,        K12.row,        K21.row + n,     K22.row + n]).astype(np.int32, copy=False)
        cols = np.concatenate([K11.col,        K12.col + n,    K21.col,         K22.col + n]).astype(np.int32, copy=False)
        data = np.concatenate([K11.data,       K12.data,       K21.data,        K22.data]).astype(np.float64, copy=False)

        # Keep UPPER triangle only
        mask = rows <= cols
        irn = rows[mask]
        jcn = cols[mask]
        a   = data[mask]

        # 1-based indices for Fortran
        irn = irn + 1
        jcn = jcn + 1

        # Sanity: indices must be in [1, N]
        if irn.min() < 1 or jcn.min() < 1 or irn.max() > N or jcn.max() > N:
            bad = np.where((irn < 1) | (jcn < 1) | (irn > N) | (jcn > N))[0][:5]
            raise ValueError(f"MUMPS index out of range. N={N}, "
                             f"min(irn)={irn.min()}, max(irn)={irn.max()}, "
                             f"min(jcn)={jcn.min()}, max(jcn)={jcn.max()}, "
                             f"first bad idxs={bad}")

        # Sort pattern (some builds are picky)
        order = np.lexsort((jcn, irn))
        irn = np.ascontiguousarray(irn[order], dtype=np.int32)
        jcn = np.ascontiguousarray(jcn[order], dtype=np.int32)
        a   = np.ascontiguousarray(a[order],   dtype=np.float64)

        # Context
        ctx = DMumpsContext(sym=2, comm=_MUMPS_COMM) if _MUMPS_COMM is not None else DMumpsContext(sym=2)
        if hasattr(ctx, "set_silent"):
            try: ctx.set_silent()
            except Exception: pass

        # Explicit shape helps for rows/cols API
        if hasattr(ctx, "set_shape"):
            try: ctx.set_shape(int(N))
            except Exception: pass

        try:
            # Preferred API: rows/cols first -> analyze; then values -> factor
            if hasattr(ctx, "set_centralized_assembled_rows_cols") and hasattr(ctx, "set_centralized_assembled_values"):
                ctx.set_centralized_assembled_rows_cols(irn, jcn)
                # analyze
                try:
                    ctx.analyze()
                except AttributeError:
                    ctx.run(job=1)
                # values + factor
                ctx.set_centralized_assembled_values(a)
                try:
                    ctx.factor()
                except AttributeError:
                    ctx.run(job=2)
                self._api_mode = "rowscols+values"
            else:
                # Fallback API: assembled(irn,jcn,a[,N]) then analyze/factor
                try:
                    ctx.set_centralized_assembled(irn, jcn, a)
                except TypeError:
                    ctx.set_centralized_assembled(irn, jcn, a, int(N))
                try:
                    ctx.analyze(); ctx.factor()
                except AttributeError:
                    ctx.run(job=1); ctx.run(job=2)
                self._api_mode = "assembled"
        except Exception as e:
            try: ctx.destroy()
            except Exception: pass
            # If MUMPS raised -16, add a hint
            msg = str(e)
            hint = " (Hint: ensure set_shape(N), 1-based sorted indices, and only one triangle.)"
            raise RuntimeError(msg + hint) from None

        self.ctx = ctx
        self.n = n
        self.m = m

    def solve(self, rhs1, rhs2):
        import numpy as np
        rhs = np.concatenate([rhs1, rhs2]).astype(np.float64, copy=False)
        if hasattr(self.ctx, "solve"):
            self.ctx.solve(rhs)  # in-place
        elif hasattr(self.ctx, "set_rhs") and hasattr(self.ctx, "get_rhs"):
            self.ctx.set_rhs(rhs); self.ctx.run(job=3); rhs = self.ctx.get_rhs()
        else:
            raise RuntimeError("This pymumps build exposes no RHS solve API (no solve(); no set_rhs/get_rhs).")
        return rhs[:self.n].copy(), rhs[self.n:].copy()


    def __del__(self):
        try: self.ctx.destroy()
        except Exception: pass
    
class PetscKKTSolver:
    is_kkt = True
    def __init__(self, P, A, rho=1.0, beta=5.0):
        import scipy.sparse as sp, numpy as np
        P = (P if sp.isspmatrix_csc(P) else sp.csc_matrix(P)).astype(np.float64)
        A = (A if A is None or sp.isspmatrix_csc(A) else sp.csc_matrix(A))
        n = P.shape[0]; m = 0 if A is None else A.shape[0]
        I_n = sp.identity(n, format="csc", dtype=np.float64)
        K11 = (P + rho*I_n).tocsc()
        if m:
            I_m = sp.identity(m, format="csc", dtype=np.float64)
            K12 = A.T.tocsc(); K21 = A.tocsc()
            K22 = (-(1.0/beta) * I_m).tocsc()
            KKT = sp.bmat([[K11, K12],[K21, K22]], format="csr")
        else:
            KKT = K11.tocsr()
        indptr, indices, data = KKT.indptr.astype(np.int32), KKT.indices.astype(np.int32), KKT.data.astype(np.float64)
        self.N = n + m; self.n = n

        self.A = PETSc.Mat().createAIJ(size=(self.N, self.N), csr=(indptr, indices, data))
        self.A.assemble()
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.A)
        self.ksp.setType('preonly')
        pc = self.ksp.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps')
        self.ksp.setFromOptions(); self.ksp.setUp()

    def solve(self, rhs1, rhs2):
        import numpy as np
        rhs = rhs1 if rhs2 is None or rhs2.size == 0 else np.concatenate([rhs1, rhs2])

        # Build b without setArray(copy=...)
        b = PETSc.Vec().createSeq(self.N)
        b.setValues(range(self.N), rhs)
        b.assemble()

        x = PETSc.Vec().createSeq(self.N)
        self.ksp.solve(b, x)
        arr = x.getArray()
        return np.array(arr[:self.n]), np.array(arr[self.n:])
    

def admm_solve(P, q, *, A=None, b=None, prox_z=None,
               rho=1.0, beta=5.0, alpha=1.6,
               max_iter=300, rtol=1e-4, atol=1e-6, verbose=False, warm=None,
               profile=True, profile_interval=25, print_header=True, solver="auto",
               lin_solver=None):

    """
    Min 0.5 x^T P x + q^T x + sum_i g_i(z_i)
    s.t. A x = b,  x = z

    prox_z(v, t) must return prox_{t * g}(v).
    """
    P = _csc(P); q = np.asarray(q, float).ravel()
    n = q.size
    if A is not None:
        A = _csc(A); b = np.asarray(b, float).ravel(); m = b.size
    else:
        m = 0; b = np.zeros(0)

    # state
    if warm is None:
        x = np.zeros(n); z = np.zeros(n); u = np.zeros(n); y = np.zeros(m)
    else:
        x = warm.get("x", np.zeros(n)).copy()
        z = warm.get("z", np.zeros(n)).copy()
        u = warm.get("u", np.zeros(n)).copy()
        y = warm.get("y", np.zeros(m)).copy()

    # --- profiling state ---
    t_total_start = time.perf_counter()
    t_factor = 0.0
    t_x = 0.0       # cumulative x-solve time
    t_z = 0.0       # cumulative prox time
    t_dual = 0.0    # cumulative dual-update time
    last_iter_time = 0.0
    it0 = 0

    # system / process info
    n_cores = os.cpu_count() or multiprocessing.cpu_count()
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc and profile:
        try: proc.cpu_percent(None)  # prime the sampler
        except Exception: pass

    # header
    if print_header and profile:
        print("=== ADMM run info ===")
        print(f"Host cores available : {n_cores}")
        tp = _tp_summary()
        if tp:
            for row in tp:
                print(f"Threadpool         : {row['api']} ({row['prefix']}) "
                    f"n_threads={row['n_threads']} lib={row['lib']}")
        else:
            print("Threadpool         : [threadpoolctl not available]")
        print(f"Env threads        : {_print_thread_env()}")
        if proc:
            mem = _bytes_h(proc.memory_info().rss)
            print(f"Process mem (start): {mem}, threads={proc.num_threads()}")
        print("----------------------")


    def _choose_solver(P, A, rho, beta, mode):
        # ---- explicit modes first ----
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
            if not _HAVE_PETSC:
                raise RuntimeError("petsc4py not available; conda install -c conda-forge petsc petsc4py mumps-mpi mpi4py scotch metis")
            return PetscKKTSolver(P, A, rho=rho, beta=beta)
        if mode == "kkt_mumps":
            # only allow if the wrapper truly supports RHS
            _can_rhs = False
            if DMumpsContext is not None:
                try:
                    _tmp = DMumpsContext(sym=2, comm=_MUMPS_COMM) if _MUMPS_COMM is not None else DMumpsContext(sym=2)
                    _can_rhs = (hasattr(_tmp, "solve") or (hasattr(_tmp, "set_rhs") and hasattr(_tmp, "get_rhs")))
                    try: _tmp.destroy()
                    except Exception: pass
                except Exception:
                    _can_rhs = False
            if not _can_rhs:
                raise RuntimeError("This pymumps build cannot accept RHS (no solve()/set_rhs/get_rhs).")
            return MumpsKKTSolver(P, A, rho=rho, beta=beta)

        # ---- auto policy ----
        if A is None or A.shape[0] == 0:
            return CholmodNormalSolver(P, A, rho=rho, beta=beta,
                                    cholmod_mode="supernodal",
                                    ordering_method="best") if cholmod_cholesky else NormalEqSolver(P, A, rho=rho, beta=beta)

        m, n = A.shape[0], P.shape[0]
        if m and (m <= 2000 and m <= n // 4):
            return WoodburySolver(P, A, rho=rho, beta=beta)

        # prefer PETSc-KKT when available; else CHOLMOD; else SuperLU KKT
        if _HAVE_PETSC:
            return PetscKKTSolver(P, A, rho=rho, beta=beta)
        if cholmod_cholesky is not None:
            return CholmodNormalSolver(P, A, rho=rho, beta=beta,
                                    cholmod_mode="supernodal",
                                    ordering_method="best")
        return KKTSolver(P, A, rho=rho, beta=beta)



    t0 = time.perf_counter()
    if lin_solver is None:
        lin = _choose_solver(P, A, rho, beta, solver)  # your helper or NormalEqSolver(...)
    else:
        lin = lin_solver
    t_factor = time.perf_counter() - t0

    def objective(x_, z_):
        return 0.5 * x_ @ (P @ x_) + q @ x_  # (we don’t try to evaluate sum g_i here)

    for k in range(1, max_iter+1):
        t_iter0 = time.perf_counter()
        # x-step: (P + rho I + beta A^T A) x = rho(z - u) - q + A^T (beta b - y)
        if getattr(lin, "is_kkt", False):
            rhs1 = rho*(z - u) - q
            rhs2 = b - (1.0/beta)*y if m else np.zeros(0)
            tx0 = time.perf_counter()
            x, lam = lin.solve(rhs1, rhs2)
            tx_i = time.perf_counter() - tx0
        else:
            rhs = rho*(z - u) - q
            if m:
                rhs = rhs + A.T @ (beta*b - y)
            tx0 = time.perf_counter()
            x = lin.solve(rhs)
            tx_i = time.perf_counter() - tx0
        t_x += tx_i

        # --- z prox
        t = alpha*x + (1 - alpha)*z
        v = t + u
        tz0 = time.perf_counter()
        z = prox_z(v, 1.0/rho)   # IMPORTANT: t = 1/rho
        t_z += time.perf_counter() - tz0

        # --- duals
        td0 = time.perf_counter()
        u = u + (t - z)
        if m: y = y + beta*(A @ x - b)
        t_dual += time.perf_counter() - td0

        # --- residuals
        pri = np.linalg.norm(x - z)
        dua = rho * np.linalg.norm(z - v)     # proxy
        eqr = np.linalg.norm(A @ x - b) if m else 0.0

        eps_pri = atol*np.sqrt(n) + rtol*max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dua = atol*np.sqrt(n) + rtol*np.linalg.norm(u)
        eq_tol  = (atol*np.sqrt(m) + rtol*np.linalg.norm(b)) if m else 0.0

        last_iter_time = time.perf_counter() - t_iter0

        if verbose and (k == 1 or k % 25 == 0):
            print(f"iter {k:4d}  pri={pri:.2e} dua={dua:.2e} eq={eqr:.2e} "
                f"iter_t={last_iter_time*1e3:6.1f} ms")

        # --- periodic live stats (record only, no printing)
        if profile and (k == 1 or k % profile_interval == 0):
            total_elapsed = time.perf_counter() - t_total_start
            iters_done = k - it0 if it0 else k
            rate = iters_done / max(1e-12, total_elapsed)
            if proc:
                try:
                    p_cpu = proc.cpu_percent(None)  # % since last call
                    p_mem = proc.memory_info().rss
                    p_thr = proc.num_threads()
                    # Stats recorded but not printed
                except Exception:
                    pass
            it0 = k; t_x = t_z = t_dual = 0.0   # reset window counters

        if pri <= eps_pri and dua <= eps_dua and (eqr <= eq_tol):
            break

    t_total = time.perf_counter() - t_total_start
    if profile:
        print("=== ADMM summary ===")
        print(f"iters={k}  total={t_total:.3f}s  factor={t_factor:.3f}s  "
            f"last_iter={last_iter_time*1e3:.1f} ms")
        if proc:
            mem = _bytes_h(proc.memory_info().rss)
            print(f"final mem={mem}  threads={proc.num_threads()}")

    return {
        "x": x, "z": z, "u": u, "y": y,
        "iters": k, "pri_res": pri, "dual_res": dua, "eq_res": eqr,
        "stats": {
            "total_time": t_total,
            "factor_time": t_factor,
            "last_iter_time": last_iter_time,
            "x_time": t_x, "z_time": t_z, "dual_time": t_dual,   # NEW
            "cores": n_cores,
            "threadpools": _tp_summary(),
            "env_threads": _print_thread_env(),
        },
    }




