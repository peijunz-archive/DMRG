import ETH.optimization as opt
from DMRG.Ising import Hamilton_XZ
import scipy.linalg as la
import numpy as np

def min_expect(rho, H):
    w1, v = la.eigh(rho)
    w2, v = la.eigh(H)
    return np.dot(sorted(w1), sorted(w2, reverse=True))

def test_local_optimization():
    n = 2
    H = Hamilton_XZ(n)['H']
    H2 = H@H
    rho = np.diag(np.arange(H2.shape[0]) + 1)
    V = np.einsum('jk, li->ijkl', rho, H2)
    U = opt.optimize_quadratic(V, nit=100)
    val2 = np.trace(U@rho@U.T.conj()@H2).real
    assert abs(min_expect(rho, H) - val2 < 1e-6)
