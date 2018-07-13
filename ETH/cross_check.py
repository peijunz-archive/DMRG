import numpy as np
from .layers.layers_dense import LayersDense
from .layers.layers_mpo import LayersMPO, MPO_TL
from .basic import trace2
from functools import reduce
if __name__ == "__main__":
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_TL
    n = 8   # dof = 2**(2n) = 64
    d = 7   # At least 2**(2(n-2))
    rho = Rho.rho_even(n, n/2, amp=0.1, rs=np.random)
    H = Hamilton_TL(n, 1, 1, 1)['H']
    mpos = MPO_TL(1,1,1)
    L = LayersMPO(rho, mpos[0], d, n-1, mpos[1], offset=0)
    Y = LayersDense(Rho.product_rho(rho), H, D=d)
    R = Y.contract_rho()
    print(trace2(R, H).real)
    print(L.contract_all(L.H).real)
    h = [np.einsum('ijkl, lk->ij', L.H, r) for r in rho]
    #print(reduce(np.matmul, h)[0,-1].real)
    #h = [np.einsum('ijkl, lk->ij', L.H, r.conj()) for r in rho]
    #print(reduce(np.matmul, h)[0,-1].real)
    
