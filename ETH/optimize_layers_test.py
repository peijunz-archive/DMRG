from ETH import Rho
from ETH.optimize_layers import Layers
from DMRG.Ising import Hamilton_XZ, Hamilton_XX
import ETH.optimization as opt
import numpy as np
import scipy.linalg as la

def test_decrease():
    '''Test function are really optimized'''
    n = 5   # dof = 2**(2n) = 64
    d = 3   # At least 2**(2(n-2))
    H = Hamilton_XZ(n)['H']
    rho = Rho.rho_prod_even(n, n*0.7, amp=0, rs=np.random)
    print(la.eigvalsh(rho))
    mins = opt.exact_min_varE(H, rho)
    Y = Layers(rho, H, D=d)
    last = np.inf
    for i in range(10):
        l = Y.minimizeVar_cycle(forward=True)
        print(l)
        assert all(l[:-1]+1e-6>=l[1:])
        assert l[0] <= last +1e-6
        last = l[-1]
    print("Final optimized value {} close to {}".format(l[-1], mins))#opt.closeto(mins, l[-1])))

def test_for_back_symmetry():
    pass

test_decrease()
