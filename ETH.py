import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt

from spin import sigma
from functools import reduce, partial
from numpy.random import RandomState
from Ising import Hamilton_XX
import Rho
import Gibbs

'''
# Note Sep. 14
+ Done. Use Random State to seed and generate Random Hamiltonian/States
+ TODO Compare both partial $\rho$ and total $\rho$ of Gibbs and minimized one
+ Save data
+ Test the routine by eigen state with some random unitary transformation
+ Pade approximation
+ Cache some used quantity like $H^2$
+ Use iterative incremental $U=\prod e^{h_i}$
'''

trace2 = partial(np.einsum, "ij, ji")

def commuteh(h1, h2):
    '''[H1, H2]'''
    M=h1@h2
    M-=M.T.conj()
    return M

def gradient(H, rho, H2=None):
    '''The gradient for rho'=exp(ih)@rho@exp(ih)

    $$\frac{df(H)}{dx}=g'(H^2)-2g(H)g'(H)=\tr[M^2]$$

    $$\frac{d^2f(H)}{dx^2}=g''(H^2)-2g'^2(H)-2g(H)g''(H)$$
    '''
    if H2 is None:
        H2=H@H
    gh = trace2(rho, H)
    dh2 = H2-(2*gh)*H
    M = 1j*commuteh(rho, dh2)
    g1h = trace2(M, H)
    mrho = commuteh(M, rho)
    f1 = trace2(M, M)
    f2 = trace2(mrho, commuteh(M, dh2)) - 2*g1h**2
    return M, f1.real, f2.real

def Hessian(H, rho):
    '''Hessian'''
    A = sp.einsum("li,jk->ijkl", rho, H)
    rh = rho@H/2
    rh += rh.conj().T
    A += sp.einsum("ki,jm->ijmk", rh, np.eye(*rho.shape))
    A += sp.einsum("ijkl->klij", A)
    return A
    
# ============== Gibbs =====================
def minimize_rho(rho, f, df, meps=10, nit=100, err=1e-6):
    '''Add criteria for optimization'''
    cur = f(rho)
    for i in range(nit):
        M, f1, f2 = df(rho)
        if f2 <= f1/meps:
            x = meps
        else:
            x = f1/f2
        h1 = - x * M
        for i in range(10):
            U = la.expm(1j*h1)
            rho_try = U@rho@U.T.conj()
            nxt = f(rho_try)
            if nxt < cur:
                cur = nxt
                rho = rho_try
                break
            h1/=2
        if f1 < err:
            break
    return rho

def grad2(H2, rho):
    M = 1j*commuteh(rho, H2)
    f1 = trace2(M, M)
    f2 = trace2(commuteh(M, rho), commuteh(M, H2))
    return M, f1.real, f2.real

def minimize_var_fix(H, rho, E, meps=10, nit=100):
    H2 = (H-E)@(H-E)
    df = partial(grad2, H2)
    f = lambda r: trace2(H2, r).real
    return minimize_rho(rho, f, df, meps, nit)

def minimize_var_nfix(H, rho, meps=10, nit=100):
    H2=H@H
    f = lambda r: trace2(r, H2).real-trace2(r, H).real**2
    df = partial(gradient, H, H2=H2)
    return minimize_rho(rho, f, df, meps, nit)

def minimize_var(H, rho, E=None, meps=10, nit=100):
    if E is None:
        return minimize_var_nfix(H, rho, meps=meps, nit=nit)
    else:
        return minimize_var_fix(H, rho, E, meps=meps, nit=nit)

# ======== rho generation ==============

def analyse(H, rho):
    var = Rho.energy_var(H, rho)
    E = trace2(rho, H).real
    b = Gibbs.energy2beta(H, E)
    S = -trace2(rho, la.logm(rho)).real/np.log(2)
    var_eth = Gibbs.beta2var(H, b)
    diff_rho = Rho.compare(Gibbs.beta2rho(H, b), rho)
    return (S, b, var, var_eth, *diff_rho)

def MeanRes(n, nit=10, s=None):
    H4=Hamilton_XX(n, 1/2, 1/2)
    rs = RandomState(0)
    a = np.empty([nit, 4+2*n-1])
    for i in range(nit):
        rho = Rho.rotate(Rho.rand_rho_prod(n, rs, s), rs)
        rho = minimize_var(H4, rho, nit=1000)
        a[i] = analyse(H4, rho)
        print(a[i])
    return a


if __name__ == "__main__":
    a = np.empty([11, 10, 11])
    for i, s in enumerate(np.linspace(0, 4, 10+1)):
        print("Entropy S", s)
        a[i] = MeanRes(4, s=s)
    n=1
    H4=Hamilton_XX(n, 1/2, 1/2)
    Hes=Hessian(H4, Rho.rand_rho_prod(n)).reshape(4, 4)
    #for i in range(4):
        #minimize_var(H4, rotate(rand_rho_prod(n, RandomState(0), 0), rs=RandomState(i)), nit=1000)
    np.save("meanres", a)
