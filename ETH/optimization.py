import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt
from .basic import *

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
