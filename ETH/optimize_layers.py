import numpy as np
from . import optimization as opt
from .basic import trace2
from .layers.layers_dense import LayersDense

#@profile
def minimizeVarE_cycle(Y, E=0, forward=True):
    if E:
        H2 = Y.H2-(2*E)*Y.H+E**2*np.eye(*Y.H.shape)
    else:
        H2 = Y.H2
    l = []
    if forward:
        for sp, V in Y.contract_cycle(H2):
            Y[sp], varE = opt.minimize_quadratic_local(V, Y[sp])
            l.append(varE)
    else:
        for sp, V in Y.contract_cycle(H2, back=True):
            U, varE = opt.minimize_quadratic_local(V, Y[sp].T.conj())
            Y[sp] = U.T.conj()
            l.append(varE)
    return np.array(l)

def minimizeVar_cycle(Y, forward=True):
    l = []
    if forward:
        for sp, V, V2 in Y.contract_cycle(Y.H, Y.H2):
            Y[sp], var = opt.minimize_var_local(V, V2, Y[sp])
            l.append(var)
    else:
        for sp, V, V2 in Y.contract_cycle(Y.H, Y.H2, back=True):
            U, var = opt.minimize_var_local(V, V2, Y[sp].T.conj())
            Y[sp] = U.T.conj()
            l.append(var)
    return np.array(l)

def minimizeVar(Y, n=100, rel=1e-10):
    last = np.inf
    for i in range(n):
        cur = minimizeVar_cycle(Y)
        print(i, cur)
        if last-cur < rel*cur:
            break
        last=cur
    return cur

def minimize_local(H, rho, D=4, dim=2, n=100, rel=1e-6):
    Y = LayersDense(rho, H, D=D, dim=dim)
    last = trace2(rho, Y.H2).real - trace2(rho, H).real**2
    for i in range(n):
        l = minimizeVar_cycle(Y)
        print(i, l)
        if last-l[-1] < rel*l[-1]:
            break
        last=l[-1]
    print("Exit at {}".format(i))
    return Y.contract_rho()


if __name__ == "__main__":
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_XX
    n = 5   # dof = 2**(2n) = 64
    d = 2   # At least 2**(2(n-2))
    arg_tpl = {"n": n, "delta": 0.54, "g": 0.1}
    H = Hamilton_XZ(n)['H']
    #print(H)
    rho = Rho.rho_even(n, n/2, amp=0, rs=np.random)
    print(rho)
    Y = LayersDense(Rho.product_rho(rho), H, D=d)
    for i in range(10):
        print(i, minimizeVarE_cycle(Y, forward=True))
