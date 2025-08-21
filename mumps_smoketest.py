# petsc_mumps_smoketest.py
import numpy as np
from petsc4py import PETSc

# Symmetric indefinite A = [[2, 1], [1, -1]]
indptr  = np.array([0, 2, 4], dtype=np.int32)          # row 0 has 2 nnz, row 1 has 2 nnz
indices = np.array([0, 1, 0, 1], dtype=np.int32)       # (0,0),(0,1),(1,0),(1,1)
data    = np.array([2.0, 1.0, 1.0, -1.0], dtype=np.float64)

A = PETSc.Mat().createAIJ(size=(2,2), csr=(indptr, indices, data))
A.assemble()

b = PETSc.Vec().createSeq(2); b.setValues([0,1], [1.0, 0.0]); b.assemble()
x = PETSc.Vec().createSeq(2)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('preonly')
pc = ksp.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps')
ksp.setFromOptions(); ksp.setUp()

ksp.solve(b, x)
print("x =", x.getArray())         # expect ~[1/3, 1/3] ≈ [0.33333333, 0.33333333]
