# sepmi/linear_solvers.py
from __future__ import annotations
import os, multiprocessing
n = os.cpu_count() or multiprocessing.cpu_count() or 1

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, str(n))

os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
    from sksparse.cholmod import analyze
except Exception:
    cholmod_cholesky = None
    analyze = None

try:
    import mumps as _mumps_mod
    DMumpsContext = getattr(_mumps_mod, "DMumpsContext", None)
    _HAVE_MUMPS = DMumpsContext is not None
except Exception:
    DMumpsContext = None
    _HAVE_MUMPS = False

try:
    from mpi4py import MPI
    _MUMPS_COMM = MPI.COMM_SELF
except Exception:
    _MUMPS_COMM = None

try:
    from petsc4py import PETSc
    _HAVE_PETSC = True
except Exception:
    _HAVE_PETSC = False


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
        self._fact = cholmod_cholesky(
            K, mode=cholmod_mode, ordering_method=ordering_method
        )

    def solve(self, rhs):
        return self._fact(rhs)


class CholmodNormalSolverReusable:
    def __init__(self, P, rho=1.0, cholmod_mode="supernodal", ordering_method="best"):
        if analyze is None:
            raise RuntimeError("CHOLMOD unavailable (install scikit-sparse).")
        P = _csc(P).astype(np.float64)
        n = P.shape[0]
        I = sp.identity(n, format="csc", dtype=np.float64)
        self.P = P; self.I = I; self.rho = float(rho)
        self._an = analyze(P + 0*I, mode=cholmod_mode, ordering_method=ordering_method)
        self._fact = self._an.cholesky(P + self.rho*I)

    def solve(self, rhs):
        return self._fact(rhs)


class WoodburySolver:
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

        AT = A.T.astype(np.float64, copy=False).toarray(order='F')
        self.X = self._Hsolve(AT)
        S = np.eye(m, dtype=np.float64) + self.beta * (A @ self.X)
        self._S_cho = la.cho_factor(S, lower=True, check_finite=False)

    def solve(self, rhs):
        y = self._Hsolve(rhs)
        t = self.beta * (self.A @ y)
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
        self._lu = spla.splu(KKT, permc_spec="COLAMD", diag_pivot_thresh=0.0)
        self.n, self.m = n, m

    def solve(self, rhs1, rhs2):
        rhs = np.concatenate([rhs1, rhs2])
        sol = self._lu.solve(rhs)
        return sol[:self.n], sol[self.n:]


class MumpsKKTSolver:
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

        K11 = (P + rho * I_n).tocoo()
        K12 = A.T.tocoo()
        K21 = A.tocoo()
        K22 = (-(1.0 / beta) * I_m).tocoo()

        rows = np.concatenate([K11.row,        K12.row,        K21.row + n,     K22.row + n]).astype(np.int32, copy=False)
        cols = np.concatenate([K11.col,        K12.col + n,    K21.col,         K22.col + n]).astype(np.int32, copy=False)
        data = np.concatenate([K11.data,       K12.data,       K21.data,        K22.data]).astype(np.float64, copy=False)

        mask = rows <= cols
        irn = rows[mask]
        jcn = cols[mask]
        a   = data[mask]

        irn = irn + 1
        jcn = jcn + 1

        order = np.lexsort((jcn, irn))
        irn = np.ascontiguousarray(irn[order], dtype=np.int32)
        jcn = np.ascontiguousarray(jcn[order], dtype=np.int32)
        a   = np.ascontiguousarray(a[order],   dtype=np.float64)

        ctx = DMumpsContext(sym=2, comm=_MUMPS_COMM) if _MUMPS_COMM is not None else DMumpsContext(sym=2)
        if hasattr(ctx, "set_silent"):
            try: ctx.set_silent()
            except Exception: pass

        if hasattr(ctx, "set_shape"):
            try: ctx.set_shape(int(N))
            except Exception: pass

        try:
            if hasattr(ctx, "set_centralized_assembled_rows_cols") and hasattr(ctx, "set_centralized_assembled_values"):
                ctx.set_centralized_assembled_rows_cols(irn, jcn)
                try:
                    ctx.analyze()
                except AttributeError:
                    ctx.run(job=1)
                ctx.set_centralized_assembled_values(a)
                try:
                    ctx.factor()
                except AttributeError:
                    ctx.run(job=2)
                self._api_mode = "rowscols+values"
            else:
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
            self.ctx.solve(rhs)
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
        b = PETSc.Vec().createSeq(self.N)
        b.setValues(range(self.N), rhs)
        b.assemble()
        x = PETSc.Vec().createSeq(self.N)
        self.ksp.solve(b, x)
        arr = x.getArray()
        return np.array(arr[:self.n]), np.array(arr[self.n:]) 