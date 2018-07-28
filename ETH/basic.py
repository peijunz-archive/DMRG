from functools import partial
import scipy.linalg as la
import numpy as np
from typing import Callable
trace2 = partial(np.einsum, "ij, ji")


def xlog2(x):
    """:math:`x\log_2(x)`

    Args
    -----
        x:np.ndarray
    Returns
    -----
        np.ndarray
    """
    return x * (np.log2(x+(x == 0)))


def entropy(r):
    """Entropy (unit is bit)

    Args
    -----
        r:np.ndarray
            Eigenvalues of rho
    Returns
    -----
        float:
            Entropy
    """
    r = r/np.sum(r)
    return -sum(r * (np.log2(r + (r == 0))))


def commuteh(h1, h2):
    """Find commutor for hermitian operators: [H1, H2]

    Args
    -----
        h1:np.ndarray
            Hermitian H1
        h1:np.ndarray
            Hermitian H2
    Returns
    -----
        np.ndarray:
            [H1, H2]=H1H2-H2H1
    """
    M = h1@h2
    M -= M.T.conj()
    return M


def energy_var(H, rho, H2=None):
    """Energy variance for given H and rho

    Args
    -----
        H:np.ndarray
            Hamiltonian
        rho:np.ndarray
            Density matrix
    Returns
    -----
        np.ndarray:
            <H^2> - <H>^2
    """
    if H2 is None:
        H2 = H@H
    res = trace2(H2, rho) - trace2(H, rho)**2
    return res.real


def rand_unitary(shape, amp=1, rs=np.random):
    """Generate random unitary matrix

    Args
    ---
        shape: np.ndarray
            shape of result
        amp: float
            amplitude of random rotation. The possible rotation angle
            is (0, amp*pi). 0 <= amp <=1
        rs: np.random.RandomState
            random state
    Returns
    ----
        np.ndarray:
            Random unitary
    """
    Hr, Hi = rs.randn(2, *shape)
    H = Hr + 1j * Hi
    U, *_ = la.svd(Hr + 1j * Hi)
    if amp < 1:
        return la.expm(amp * la.logm(U))
    return U


def bitsign(x):
    """
    Element-wise operation:
        + Return +1  for positive numbers
        + Return -1  for negative numbers
    """
    return 1 - (np.signbit(x) << 1)


def mlinspace(n):
    """Sample n points with same interval in (0, 1)"""
    return n, (np.arange(n)+0.5)/n


def rand_rotate(rho, amp=1, rs=np.random):
    """Rotate density matrix with randomly generated Unitary
    """
    U = rand_unitary(rho.shape, amp, rs)
    return U@rho@U.T.conj()


def verify_mini(fun:Callable, rho, x=0.01, n=100):
    """Verify the function is at a minimum rho in small range

    Args
    ----
        fun: Callable
            Function of rho
        rho: np.ndarray
            Current density
        x: float
            Range of variation
        n: int
            Number of trials
    Returns
    ----
        bool:
            True if no larger values found
    """
    bench = fun(rho)
    for i in range(n):
        test = fun(rand_rotate(rho, x))
        if test < bench:
            return False
    return True
