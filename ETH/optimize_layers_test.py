from ETH import Rho
from ETH.optimize_layers import LayersDense
from DMRG.Ising import nearest, Hamilton_XZ, Hamilton_XX
from DMRG.spin import sigma
from ETH.basic import *
from functools import reduce
import ETH.optimization as opt
import numpy as np
import scipy.linalg as la
import pytest

@pytest.mark.parametrize("n", [2, 3])#, 4])
@pytest.mark.parametrize("k", [0.3, 0.7])
def test_small_chain_varE(n, k, nit=50):
    '''Test function are really optimized'''

    tol = {2:1e-6, 3:1e-2, 4:0.1}
    H = Hamilton_XZ(n)['H']
    rho = Rho.rho_prod_even(n, n*k, rs=np.random)
    mini = opt.exact_min_varE(H, rho)
    Y = LayersDense(rho, H, D=4**(n-2)//(n-1)*2+1)
    last = np.inf
    for i in range(nit):
        l = Y.minimizeVarE_cycle()
        assert all(l[:-1]+1e-6>=l[1:])
        assert l[0] <= last +1e-6
        if last-l[-1] < tol[n]*max(l[-1], 1)/100:
            break
        last = l[-1]
    assert abs(last-mini)<3*tol[n]*max(mini, 1), "Global minimum for varE not found"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("choice", ["prod", "sum"])
def test_local_H2(n, choice, k=0.5, d=2, tol=1e-4):
    '''Test local Hamiltonian H^2'''

    if choice == "prod":
        H2 = reduce(np.kron, [np.diag([2, 1])]*n)
    else:
        H2 = nearest(n, np.diag([-1, 1]))
    rho = Rho.rho_prod_even(n, n*k, rs=np.random)
    Y = LayersDense(rho, H2=H2, D=d)
    mini = opt.min_expect(H2, rho)
    last = np.inf
    for i in range(50):
        l = Y.minimizeVarE_cycle()
        if last-l[-1] < tol*l[-1]/100:
            break
        last = l[-1]
    assert abs(last-mini)<3*tol*abs(max(mini, 1)), "Global minimum for local Hamiltonian not found"

@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("k", [0.3, 0.7])
def test_reverse_engineering(n, k, nit=50, tol=1e-4):
    '''Optimize H^2 with exact solution,
    and then displace H^2 or rho by local unitary'''
    H = Hamilton_XZ(n)['H']
    H2 = H@H
    w, v = la.eigh(H2)
    rho = np.sort(np.diag(Rho.rho_prod_even(n, n*k)))[::-1]
    rho = v@np.diag(rho)@v.T.conj()
    mini = opt.min_expect(H2, rho)
    assert abs(trace2(rho, H2) - mini)<1e-6
    Y = LayersDense(rho, H2=H2, D=3)
    for ind in Y.indexes:
        Y[ind] = rand_unitary(np.eye(4), amp=0.1)
    rho2 = Y.contract_rho()
    assert trace2(rho2, H2) > mini
    #print(trace2(rho2, H2).real, mini)
    Y = LayersDense(rho2, H2=H2, D=3)
    last = np.inf
    for i in range(1000):
        l = Y.minimizeVarE_cycle()
        if last-l[-1] < tol*l[-1]/100:
            break
        last = l[-1]
    print("Last", i, last, l)
    assert abs(last-mini)<3*tol*abs(max(mini, 1)), "Global minimum for local Hamiltonian not found"


def test_for_back_symmetry():
    '''Test function are really optimized'''
    n=8
    k = 0.5
    H = Hamilton_XZ(n)['H']
    rho = Rho.rho_prod_even(n, n*k)#, rs=np.random)
    rho2 = rand_rotate(rho)
    Y = LayersDense(rho, rho, D=3)
    Y.H2 = rho2
    Y2 = LayersDense(rho2, rho, D=3)
    Y2.H2 = rho
    for i in range(2):
        l1 = Y.minimizeVarE_cycle()
        l2 = Y2.minimizeVarE_cycle(forward=False)
        assert la.norm(l1-l2)<1e-2*la.norm(l1)
