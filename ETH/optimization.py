import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt
from .basic import *

#@profile
def gradient(H, rho, H2=None, err=0):
    '''The gradient for rho'=exp(ih)@rho@exp(ih)

    $$\frac{df(H)}{dx}=g'(H^2)-2g(H)g'(H)=\tr[M^2]$$

    $$\frac{d^2f(H)}{dx^2}=g''(H^2)-2g'^2(H)-2g(H)g''(H)$$
    '''
    if H2 is None:
        H2=H@H
    E = trace2(rho, H)
    h = H2-(2*E)*H
    M = 1j*commuteh(rho, h)
    f1m = la.norm(M)**2
    M2 = 1j*commuteh(rho, H)
    Msq = la.norm(M2)**2
    sigma = np.sqrt(energy_var(H, rho, H2))
    if f1m > 0.1*sigma*Msq:
        M_ = M
    else:
        M_ = M2
    f1m_ = trace2(M_, M)
    f2_ = trace2(commuteh(M_, rho), commuteh(M_, h)) - 2*trace2(M_, M2)**2
    #print(sigma, f1m_.real, f2_.real)
    return M_, f1m_.real, f2_.real

#@profile
def grad2(H2, rho):
    M = 1j*commuteh(rho, H2)
    f1 = la.norm(M)**2
    f2 = trace2(commuteh(M, rho), commuteh(M, H2))
    return M, f1.real, f2.real

def Hessian(H, rho):
    '''Hessian'''
    A = sp.einsum("li,jk->ijkl", rho, H)
    rh = rho@H/2
    rh += rh.conj().T
    A += sp.einsum("ki,jm->ijmk", rh, np.eye(*rho.shape))
    A += sp.einsum("ijkl->klij", A)
    return A

#@profile
def expm2_ersatz(h):
    I = np.eye(*h.shape)
    # Better version of
    #U = (I+h)@la.inv(I-h)
    U = np.linalg.solve(I-h, I+h)
    #print(U@U.T.conj())
    return U

#@profile
def minimize_rho(rho, f, df, meps=10, nit=100, err=0):
    '''Add criteria for optimization'''
    cur = np.inf
    for i in range(nit):
        M, f1, f2 = df(rho)
        if f2 <= f1/meps:
            x = -meps
        else:
            x = -f1/f2
        for j in range(10):
            #U = la.expm((-1j*x)*M)
            U = expm2_ersatz((0.5j*x)*M)
            rho_try = U@rho@U.T.conj()
            #print(rho)
            nxt = f(rho_try)
            if nxt < cur or (f1 == 0):
                #print(cur, nxt-cur)
                cur = nxt
                rho = rho_try
                break
            #print("Bad", cur, nxt-cur)
            x/=2
        if j==9 or (f1 < err and f2 >= 0):
            # Judge convergence
            print("Stop at {} with f1={}, f2={}".format(i, f1, f2))
            break
    return rho

def minimize_var_fix(H, rho, E, meps=10, nit=100, err=0):
    H2 = (H-E)@(H-E)
    f = lambda r: trace2(H2, r).real
    df = partial(grad2, H2)
    return minimize_rho(rho, f, df, meps, nit, err)

def minimize_var_nfix(H, rho, meps=10, nit=100, err=0):
    H2=H@H
    f = partial(energy_var, H, H2=H2)
    #f = lambda r: trace2(r, H2).real-trace2(r, H).real**2
    df = partial(gradient, H, H2=H2, err=err)
    return minimize_rho(rho, f, df, meps, nit, err)

def minimize_var(H, rho, E=None, meps=10, nit=100, err=0):
    if E is None:
        return minimize_var_nfix(H, rho, meps=meps, nit=nit, err=err)
    else:
        return minimize_var_fix(H, rho, E, meps=meps, nit=nit, err=err)
