import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt
from .basic import *
from functools import partial
from itertools import permutations


'''Global optimization'''


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

# @profile


def grad2(h, rho):
    M = 1j * commuteh(rho, h)
    f1 = la.norm(M)**2
    f2 = trace2(commuteh(M, rho), commuteh(M, h))
    return M, f1.real, f2.real

# @profile


def expm2_ersatz(h):
    I = np.eye(*h.shape)
    # Better version of
    #U = (I+h)@la.inv(I-h)
    U = np.linalg.solve(I - h, I + h)
    # print(U@U.T.conj())
    return U

# @profile


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
        #print(cur, nabla, hessian)
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
    # f = lambda r: trace2(r, H2).real-trace2(r, H).real**2
    df = partial(gradient, H, H2=H2, err=err)
    return minimize_rho(rho, f, df, meps, nit, err)


def minimize_var(H, rho, E=None, meps=10, n=100, rel=0):
    if E is None:
        return minimize_var_nfix(H, rho, meps=meps, nit=n, err=rel)
    else:
        return minimize_var_fix(H, rho, E, meps=meps, nit=n, err=rel)


'''Local Code'''


def improve(_grad, _f, meps):
    '''Single step of Steepest descent algorithm'''
    M, f1, f2 = _grad
    istep = 1 / meps
    orig = _f()  # np.einsum('iijj', V).real
    step = - bitsign(f1) / np.clip(f2 / np.abs(f1), istep, None)
    for i in range(4):
        U = la.expm(1j * step * M)
        new = _f(U)
        if new < orig:
            return U, new
        step /= 2
    return None, orig


def f_quadratic_local(V, U=None):
    if U is None:
        return np.einsum('iijj', V).real
    else:
        return np.einsum("ijkl, ij, kl", V, U, U.T.conj()).real


def nabla_quadratic_local(V):
    return (1j * (np.trace(V, axis1=2, axis2=3) - np.trace(V, axis1=0, axis2=1))).T


def df1_quadratic_local(M, M2=None):
    if M2 is None:
        M2 = M
    return trace2(M, M2).real
    # return np.einsum('ij, ji', nabla_V, M).real


def df2_quadratic_local(V, M):
    # f2_2 = (np.einsum('ijkk, il, lj', V, M, M) + np.einsum('kkij, il, lj', V, M, M))/2
    # but the two terms equal to each other
    f2_1 = np.einsum('ijkl, ij, kl', V, M, M).real
    f2_2 = np.einsum('ijkk, il, lj', V, M, M).real
    # /2+ np.einsum('kkij, il, lj', V, M, M).real/2
    return (f2_1 - f2_2)*2


def df_quadratic_local(V):
    M = nabla_quadratic_local(V)
    f1 = df1_quadratic_local(M)  # np.einsum('ij, ji', nabla_V, M).real
    f2 = df2_quadratic_local(V, M)
    return M, f1, f2


def minimize_quadratic_local(V, U=None, nit=10, meps=1):
    '''Optimize <H^2>'''
    if U is None:
        U = np.eye(4)
    for i in range(nit):
        du, f = improve(df_quadratic_local(
            V), partial(f_quadratic_local, V), meps)
        if du is None:
            break
        V = np.einsum('ijkl, ip, ql->pjkq', V, du, du.T.conj())
        U = U@du
    return U, f


def f_var_local(V, V2, U=None):
    return f_quadratic_local(V2, U) - f_quadratic_local(V, U)**2


def df_var_local(V, V2):
    E = f_quadratic_local(V)
    # print(E)
    M_1 = nabla_quadratic_local(V)
    M_2 = nabla_quadratic_local(V2)
    M = M_2 - (2*E)*M_1

    f1_1 = df1_quadratic_local(M, M_1)
    f1 = df1_quadratic_local(M)

    f2_1 = df2_quadratic_local(V, M)
    f2_2 = df2_quadratic_local(V2, M)
    f2 = f2_2 - (2*E)*f2_1 - 2*f1_1**2
    return M, f1, f2


def minimize_var_local(V, V2, U=None, nit=10, meps=1):
    '''minimize <H^2>-<H>^2'''
    if U is None:
        U = np.eye(4)
    for i in range(nit):
        #print(df_var_local(V, V2))
        du, f = improve(df_var_local(V, V2), partial(f_var_local, V, V2), meps)
        if du is None:
            break
        V = np.einsum('ijkl, ip, ql->pjkq', V, du, du.T.conj())
        V2 = np.einsum('ijkl, ip, ql->pjkq', V2, du, du.T.conj())
        U = U@du
    # print(f)
    return U, f


'''Exact min solutions'''


def exact_min_var(H, rho):
    if max(H.shape) > 8:
        return np.zeros(1)
    l = la.eigvalsh(rho)[::-1]
    assert all(l >= 0)
    l /= sum(l)
    E = la.eigvalsh(H)

    mins = {}
    for E2 in permutations(E):
        EE = np.array(E2)
        var = np.dot(EE**2, l)-np.dot(EE, l)**2
        dE2 = (EE - np.dot(EE, l))**2
        # if all(dE2 == sorted(dE2)):
        mins[var] = l[np.argsort(dE2[::-1])]
    return mins


def min_expect(rho, H2):
    w1 = la.eigvalsh(rho)
    w2 = la.eigvalsh(H2)
    return np.dot(sorted(w1), sorted(w2, reverse=True))


def exact_min_varE(H, rho, E=0):
    l = la.eigvalsh(rho)[::-1]
    assert all(l >= 0)
    l /= sum(l)
    E = (la.eigvalsh(H)-E)**2
    return np.dot(sorted(E), l)


def closeto(mins, x):
    keys = list(mins.keys())
    print(sorted(keys))
    return keys[np.argmin([x-i if i <= x+1e-6 else np.inf for i in keys])]
