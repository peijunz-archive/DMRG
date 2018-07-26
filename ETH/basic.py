from functools import partial
import scipy.linalg as la
import numpy as np

trace2 = partial(np.einsum, "ij, ji")


def xlog2(x):
    '''Robust x*log_2(x) that does not diverge at 0'''
    return x * (np.log2(x+(x==0)))


def entropy(r):
    '''Entropy in bit'''
    r = r/np.sum(r)
    return -sum(r * (np.log2(r + (r==0))))


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

def rand_unitary(shape, amp=1, rs=np.random):
    '''Generate random unitary matrix
    Args:
        shape   shape of 
        amp     amplitude of random rotation. The possible rotation angle
                is (0, amp*pi). 0 <= amp <=1
        rs      random state
    '''
    Hr, Hi = rs.randn(2, *shape)
    H = Hr + 1j * Hi
    U, *_ = la.svd(Hr + 1j * Hi)
    if amp < 1:
        return la.expm(amp * la.logm(U))
    return U


def bitsign(x):
    '''
    Element-wise operation:
        +1  for positive numbers
        -1  for negative numbers
    '''
    return 1 - (np.signbit(x)<<1)


def mlinspace(n):
    '''Sample n points with same interval in (0, 1)'''
    return n, (np.arange(n)+0.5)/n


def rand_rotate(rho, amp=1, rs=np.random):
    U = rand_unitary(rho.shape, amp, rs)
    return U@rho@U.T.conj()


def verify_mini(fun, rho, x=0.01, n=100):
    '''Verify the function is at a minimum rho'''
    bench = fun(rho)
    for i in range(n):
        test = fun(rand_rotate(rho, x))
        if test < bench:
            return False
    return True
