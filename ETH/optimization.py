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
        H2 = H@H
    E = trace2(rho, H)
    h = H2 - (2 * E) * H
    M = np.array([1j * commuteh(rho, h), 1j * commuteh(rho, H)])
    n = M.shape[0]
    l = la.norm(M, axis=(1, 2))
    M /= l.reshape(-1, 1, 1)
    scale = l[0]

    def _f1(m): return scale * trace2(m, M[0]).real

    def _f2(m1, m2): return trace2(commuteh(m1, rho),
                                   commuteh(m2, h)).real - _f1(m1) * _f1(m2)
    nabla = np.array([_f1(m) for m in M])
    Hessian = np.empty([n, n], dtype='double')
    for i in range(n):
        Hessian[i, i] = _f2(M[i], M[i])
        for j in range(i + 1, n):
            Hessian[i, j] = Hessian[j, i] = _f2(M[i], M[j])
    return M, nabla, Hessian

#@profile


def grad2(h, rho):
    M = 1j * commuteh(rho, h)
    f1 = la.norm(M)**2
    f2 = trace2(commuteh(M, rho), commuteh(M, h))
    return M, f1.real, f2.real

#@profile


def expm2_ersatz(h):
    I = np.eye(*h.shape)
    # Better version of
    #U = (I+h)@la.inv(I-h)
    U = np.linalg.solve(I - h, I + h)
    # print(U@U.T.conj())
    return U

#@profile


def minimize_rho(rho, f, df, meps=0.5, nit=100, err=0):
    '''Add criteria for optimization'''
    istep = 1 / meps
    cur = np.inf
    for i in range(nit):
        M, nabla, hessian = df(rho)
        w, v = la.eigh(hessian)
        nabla = v.T.conj()@nabla
        step = - bitsign(nabla) / np.clip(w / np.abs(nabla), istep, None)
        step = v@step
        M_opt = np.einsum('i, ijk', step, M)
        f1 = la.norm(nabla)
        convex = all(w >= 0)
        for j in range(10):
            U = expm2_ersatz((1j / 2**j) * M_opt)
            rho_try = U@rho@U.T.conj()
            nxt = f(rho_try)
            if nxt < cur or (f1 == 0):
                cur = nxt
                rho = rho_try
                break
        if ((i * 10 > nit) and j == 9):  # or (f1 < err and convex):
            break
    print("Stop at {} with f={}, f1={}, convex={}".format(i, cur, f1, convex))
    return rho


def minimize_var_fix(H, rho, E, meps=10, nit=100, err=0):
    Delta = H - E * np.eye(*H.shape)
    h = Delta@Delta

    def f(r): return trace2(h, r).real
    df = partial(grad2, h)
    return minimize_rho(rho, f, df, meps, nit, err)


def minimize_var_nfix(H, rho, meps=10, nit=100, err=0):
    H2 = H@H
    f = partial(energy_var, H, H2=H2)
    #f = lambda r: trace2(r, H2).real-trace2(r, H).real**2
    df = partial(gradient, H, H2=H2, err=err)
    return minimize_rho(rho, f, df, meps, nit, err)


def minimize_var(H, rho, E=None, meps=10, nit=100, err=0):
    if E is None:
        return minimize_var_nfix(H, rho, meps=meps, nit=nit, err=err)
    else:
        return minimize_var_fix(H, rho, E, meps=meps, nit=nit, err=err)


def improve(V, meps):
    M = 1j * (np.trace(V, axis1=0, axis2=1) - np.trace(V, axis1=2, axis2=3))
    f1 = np.einsum('ij, ji', M, M).real
    # f2_2 = (np.einsum('ijkk, il, lj', V, M, M) + np.einsum('kkij, il, lj', V, M, M))/2
    # but he two terms equal to each other
    f2_1 = np.einsum('ijkl, ij, kl', V, M, M).real
    f2_2 = np.einsum('ijkk, il, lj', V, M, M).real
    f2 = f2_1 - f2_2
    istep = 1 / meps
    orig = np.einsum('iijj', V).real
    step = - bitsign(f1) / np.clip(f2 / np.abs(f1), istep, None)
    for i in range(4):
        U = la.expm(1j * step * M)
        new = np.einsum("ijkl, ij, kl", V, U, U.T.conj()).real
        if new < orig:
            return np.einsum('ijkl, ip, ql->pjkq', V, U, U.T.conj()), U
        step /= 2
    return V, None


def optimize_quadratic(V, U=np.eye(4), nit=10, meps=1):
    '''Optimize '''
    for i in range(nit):
        V, du = improve(V, meps)
        if du is None:
            break
        U = U@du
    return U
