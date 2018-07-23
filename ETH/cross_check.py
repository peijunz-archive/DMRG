import numpy as np
from .layers.layers_dense import LayersDense
from .layers.layers_mpo import LayersMPO, MPO_TL, ud2rl
from .basic import trace2, rand_unitary
from numpy.random import RandomState
from functools import reduce
if __name__ == "__main__":
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_TL
    n = 9   # dof = 2**(2n) = 64
    d = 6   # At least 2**(2(n-2))
    rs = np.random#RandomState(123581321)
    rho = Rho.rho_even(n, n/2, amp=0.1, rs=rs)
    H = Hamilton_TL(n, 1, 1, 1)['H']
    mpos = MPO_TL(1,1,1)
    L = LayersMPO(rho, mpos[0], d, n-1, mpos[1], offset=0)
    Y = LayersDense(Rho.product_rho(rho), H, D=d)
    for i in Y.indices:
        print(i)
        Y[i] = rand_unitary(np.eye(4), rs=rs)
        L[i[::-1]] = ud2rl(Y[i].T.conj())
    R = Y.contract_rho()
    print(trace2(R, H@H).real)
    for i, l, r in L.sweep(L.H2, 1):
        l = L.apply_pair(i, l, L.H2)
        print(np.dot(l.flatten(), r.flatten()).real)
    #h = [np.einsum('ijkl, lk->ij', L.H2, r) for r in rho]
    #print(reduce(np.matmul, h)[0,-1].real)
    #h = [np.einsum('ijkl, lk->ij', L.H, r.conj()) for r in rho]
    #print(reduce(np.matmul, h)[0,-1].real)

