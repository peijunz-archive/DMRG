from functools import partial
import scipy.linalg as la
import numpy as np

trace2 = partial(np.einsum, "ij, ji")


@np.vectorize
def xlog2x(x):
    '''Robust x*log_2(x) that does not diverge at 0'''
    if x > 0:
        return x * np.log2(x)
    else:
        return 0.


def entropy(r):
    '''Entropy in bit'''
    return -sum(xlog2x(r / np.sum(r)))


def commuteh(h1, h2):
    '''Find commutor for hermitian operators: [H1, H2]'''
    M = h1@h2
    M -= M.T.conj()
    return M


def energy_var(H, rho, H2=None):
    '''Energy variance for given H and rho'''
    if H2 is None:
        H2 = H@H
    res = trace2(H2, rho) - trace2(H, rho)**2
    return res.real


def rand_unitary(rho, amp=None, rs=np.random):
    Hr, Hi = rs.randn(2, *rho.shape)
    H = Hr + 1j * Hi
    U, _, _ = la.svd(Hr + 1j * Hi)
    if amp is not None:
        return la.expm(amp * la.logm(U))
    return U


def bitsign(x):
    return 1 - 2 * np.signbit(x)


def mlinspace(n):
    return (np.arange(n)+0.5)/n


def rand_rotate(rho, amp=None, rs=np.random):
    U = rand_unitary(rho, amp, rs)
    return U@rho@U.T.conj()


def verify_mini(fun, rho, x=0.01, n=100):
    bench = fun(rho)
    for i in range(n):
        test = fun(rand_rotate(rho, x))
        if test < bench:
            return False
    return True
