import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt
from .basic import *
from functools import partial
from itertools import permutations

###
#
# Global optimization
#
###

def grad_single(h, rho):
    """gradient of tr[U rho U^+ h], U=exp(ixM)

    Please refer to "doc/global_optimization"

    Args
    ---
        h: np.ndarray
            Hermitian operator
        rho: np.ndarray
            Density matrix
    Returns
    ---
        M: np.ndarray
            Gradient, the direction of optimization. M = i[rho, h]
        f': np.float
            First order derivative in M direction
        f'': np.float
            Second order derivative in M direction
    """
    M = 1j * commuteh(rho, h)
    f1 = la.norm(M)**2
    f2 = trace2(commuteh(M, rho), commuteh(M, h))
    return M, f1.real, f2.real

def gradient_simple(H, rho, H2=None):
    """gradient of tr[U rho U^+ H^2] - tr[U rho U^+ H]^2, U=exp(ixM)

    Please refer to "doc/global_optimization"

    Args
    ---
        h: np.ndarray
            Hermitian operator
        rho: np.ndarray
            Density matrix
        H2: np.ndarray, optional
            H^2
    Returns
    ---
        M: np.ndarray
            Gradient, the direction of optimization, M = i[rho, (H-E)^2]
        f': np.float
            First order derivative in M direction
        f'': np.float
            Second order derivative in M direction
    """
    if H2 is None:
        H2=H@H
    energy = trace2(rho, H)
    dH2 = H2-(2*energy)*H
    M = 1j*commuteh(rho, dH2)
    f1 = trace2(M, M)
    f2 = trace2(commuteh(M, rho), commuteh(M, dH2)) - 2*trace2(M, H)**2
    return M, f1.real, f2.real

def gradient_compound(H, rho, H2=None):
    """gradient of tr[U rho U^+ H^2] - tr[U rho U^+ H]^2, U=exp(ixM), plus
    i[rho, H] as the second direction to avoid saddle points

    Please refer to "doc/global_optimization"

    Args
    ---
        H: np.ndarray
            Hermitian operator
        rho: np.ndarray
            Density matrix
        H2: np.ndarray, optional
            H^2
    Returns
    ---
        M: np.ndarray
            Directions of optimization: M_0 = i[rho, (H-E)^2], M_1 = i[rho, H]
        nabla: np.ndarray
            Vector pf first order derivatives
        Hessian: np.ndarray
            Hessian Matrix
    """
    if H2 is None:
        H2 = H@H
    E = trace2(rho, H)
    h = H2 - (2 * E) * H
    M = np.array([1j * commuteh(rho, h), 1j * commuteh(rho, H)])
    l = la.norm(M, axis=(1, 2))
    M /= l.reshape(-1, 1, 1)
    scale = l[0]

    def _f1(m): return scale * trace2(m, M[0]).real

    def _f2(m1, m2): return trace2(commuteh(m1, rho),
                                   commuteh(m2, h)).real - _f1(m1) * _f1(m2)
    nabla = np.array([_f1(m) for m in M])
    n = M.shape[0]
    Hessian = np.empty([n, n], dtype='double')
    for i in range(n):
        Hessian[i, i] = _f2(M[i], M[i])
        for j in range(i + 1, n):
            Hessian[i, j] = Hessian[j, i] = _f2(M[i], M[j])
    return M, nabla, Hessian


def minimize_rho(rho, f, df, max_step=0.5, nit=100):
    """Minimize observable function f of rho, by applying unitaries to rho

    Args
    ----
        rho: np.ndarray
            Density Matrix
        f: Callable[[np.ndarray], float]
        df: Callable[[np.ndarray], [np.ndarray, float, float]]
            It returns optimization directions, f'(or gradients), f''(or Hessian)
        max_step: float
            Max step size for optimization
        nit: int
            Number of optimization cycles
        err: float
    Returns
    ----
        rho: np.ndarray
            "Optimal" Density matrix
    """
    istep = 1 / max_step
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
            U = expm_((1j / 2**j) * M_opt)
            rho_try = U@rho@U.T.conj()
            nxt = f(rho_try)
            if nxt < cur or (f1 == 0):
                cur = nxt
                rho = rho_try
                break
        #print(cur, nabla, hessian)
        if ((i * 10 > nit) and j == 9):
            break
    print("Stop at {} with f={}, f1={}, convex={}".format(i, cur, f1, convex))
    return rho


def minimize_var_displace(H, rho, E, max_step=10, nit=100):
    """Minimize variance with fixed energy displacement"""
    Delta = H - E * np.eye(*H.shape)
    h = Delta@Delta

    def f(r): return trace2(h, r).real
    df = partial(grad_single, h)
    return minimize_rho(rho, f, df, max_step, nit, err)


def minimize_variance(H, rho, max_step=10, nit=100):
    """Minimize variance without fixing energy"""
    H2 = H@H
    f = partial(energy_var, H, H2=H2)
    # f = lambda r: trace2(r, H2).real-trace2(r, H).real**2
    df = partial(gradient_compound, H, H2=H2, err=err)
    return minimize_rho(rho, f, df, max_step, nit, err)


def minimize_var(H, rho, E=None, max_step=10, n=100, rel=0):
    """Minimize variance

    Args
    ---
        H: np.ndarray
            Hermitian observable
        rho: np.ndarray
            density matrix
        E:
            None or fixed energy displacement
        max_step: np.float
        n: int
        rel:
    """
    if E is None:
        return minimize_variance(H, rho, max_step=max_step, nit=n, err=rel)
    else:
        return minimize_var_displace(H, rho, E, max_step=max_step, nit=n, err=rel)


"""Local Code"""


def improve(_grad, _f, max_step):
    """Single step of Steepest descent algorithm"""
    M, f1, f2 = _grad
    istep = 1 / max_step
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


def minimize_quadratic_local(V, U=None, nit=10, max_step=1):
    """Optimize <H^2>"""
    if U is None:
        U = np.eye(4)
    for i in range(nit):
        du, f = improve(df_quadratic_local(
            V), partial(f_quadratic_local, V), max_step)
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


def minimize_var_local(V, V2, U=None, nit=10, max_step=1):
    """minimize <H^2>-<H>^2"""
    if U is None:
        U = np.eye(4)
    for i in range(nit):
        #print(df_var_local(V, V2))
        du, f = improve(df_var_local(V, V2), partial(f_var_local, V, V2), max_step)
        if du is None:
            break
        V = np.einsum('ijkl, ip, ql->pjkq', V, du, du.T.conj())
        V2 = np.einsum('ijkl, ip, ql->pjkq', V2, du, du.T.conj())
        U = U@du
    # print(f)
    return U, f


"""Exact min solutions"""


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
