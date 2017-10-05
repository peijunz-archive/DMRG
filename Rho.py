import numpy as np
import scipy.optimize as opt
from functools import reduce
import scipy.linalg as la
from ETH import trace2


@np.vectorize
def xlogx(x):
    if x>0:
        return x*np.log2(x)
    else:
        return 0.

def S(r, eps=1e-20):
    t=r.copy()
    t/=np.sum(t)
    return -sum(xlogx(t))

def _rho_entropy_rough(r, s):
    s0 = S(r)
    s1 = s0
    while (s0-s)*(s1-s)>0:
        if s1 < s:
            r=np.sqrt(r)
        else:
            r=r**2
        r/=r.max()
        s1 = S(r)
    return r

def rho_entropy(r, s, err=1e-5):
    smax = np.log2(len(r))
    assert(0<=s<=smax)
    r = _rho_entropy_rough(r, s)
    #if s <= err:
        #return r/np.sum(r)
    #if s >= smax-err:
        #return r/np.sum(r)
    def _ds(n):
        t=r**n
        return s-S(t)
    if _ds(0.5)*_ds(2) < 0:
        n = opt.bisect(_ds, 0.5, 2)
        r=r**n
    return r/np.sum(r)


def product_rho(L, s):
    '''Generate rho by product of List of small ones'''
    rho=reduce(np.kron, L)
    rho/=sum(rho)
    if s is None:
        return np.diag(rho)
    else:
        return np.diag(rho_entropy(rho, s))

def rand_rho(n, rs=None, s=None):
    '''Generate huge diagonal rho'''
    if rs is None:
        rs=RandomState(0)
    rho = abs(rs.randn(2**n))
    rho /= sum(rho)
    if s is None:
        return np.diag(rho)
    else:
        return np.diag(entropy.rho(rho, s))

def rand_rho_prod(n, rs=None, s=None):
    if rs is None:
        rs=RandomState(0)
    rhol = abs(rs.randn(n, 2)+1)
    rhol /= np.sum(rhol, axis=1)[:, np.newaxis]
    s= product_rho(rhol, s)
    return s

def compare(rho1, rho2):
    '''Compare density matrix by trace out different degree
    For n sites, it may trace out 0, 1,..., n-1 sites
    '''
    n=int(np.round(np.log2(rho1.shape[0])))
    diff_rho = np.empty(2*n-1)
    diff_rho[n-1] = la.norm(rho1-rho2)/la.norm(rho1+rho2)
    for i in range(1, n):
        sh = (2**i, 2**(n-i))*2
        t1, t2 = rho1.reshape(sh), rho2.reshape(sh)
        r1 = np.trace(t1, axis1=1, axis2=3)
        r2 = np.trace(t2, axis1=1, axis2=3)
        diff_rho[i-1] = la.norm(r1-r2)/la.norm(r1+r2)
        r1 = np.trace(t1, axis1=0, axis2=2)
        r2 = np.trace(t2, axis1=0, axis2=2)
        diff_rho[n+i-1] = la.norm(r1-r2)/la.norm(r1+r2)
    return diff_rho

def energy_var(H, rho, H2=None):
    '''Energy variance for given H and rho'''
    if H2 is None:
        H2=H@H
    res=trace2(H2, rho)-trace2(H, rho)**2
    #print("Var", trace2(H2, rho), trace2(H, rho))
    return res.real


def rotate(rho, rs):
    Hr, Hi = rs.randn(2, *rho.shape)
    H = Hr + 1j*Hi
    H += H.T.conj()
    U = la.expm(5j*H)
    return U@rho@U.T.conj()
