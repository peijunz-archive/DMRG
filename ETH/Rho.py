import numpy as np
import scipy.optimize as opt
from functools import reduce
import scipy.linalg as la
from numpy.random import RandomState
from .basic import *


def adjust_entropy(r, s, err=1e-5):
    '''Set entropy of rho by adjusting temperature'''
    smax = np.log2(len(r))
    assert(0 <= s <= smax)

    def f(n): return s - entropy(r**n)
    s0 = f(r)
    while s0 * f(r) > 0:
        r = np.sqrt(r) if s1 < s else r**2
        r /= r.max()
    if f(0.5) * f(2) < 0:
        r = r**opt.bisect(f, 0.5, 2)
    return r / np.sum(r)


def rand_rho(n, rs=None, s=None):
    ''' DEPRECATED '''
    '''Generate huge diagonal rho'''
    if rs is None:
        rs = RandomState(0)
    rho = abs(rs.randn(2**n))
    rho /= sum(rho)
    if s is None:
        return np.diag(rho)
    else:
        return np.diag(rho_entropy(rho, s))


def rand_rho_prod(n, rs=None, s=None):
    ''' DEPRECATED '''
    if rs is None:
        rs = RandomState(0)
    rhol = abs(rs.randn(n, 2) + 1)
    rhol /= np.sum(rhol, axis=1)[:, np.newaxis]
    s = product_rho(rhol, s)
    return s


def product_rho(L, s=None):
    '''Generate rho by product of List of small ones'''
    rho = reduce(np.kron, L)
    rho /= sum(rho)
    if s is None:
        return np.diag(rho)
    else:
        return np.diag(rho_entropy(rho, s))


def rho_prod_even(n, s=0, amp=None, rs=None):
    s1 = s / n
    err = 1e-10
    if s1 < err:
        x = 0.
    elif s1 > 1 - err:
        x = .5
    else:
        x = opt.bisect(lambda x: s1 + x * np.log2(x) + (1 - x)
                       * np.log2(1 - x) if x > 0 else s1, err, .5 - err)
    if rs:
        rho = reduce(np.kron, [rand_rotate(np.diag([x, 1-x]), amp, rs=rs) for i in range(n)])
        return rho
    else:
        return product_rho([[x, 1 - x]] * n)

def compare_segm(r1, r2, start, end):
    if start >= end:
        return np.nan
    n = int(np.round(np.log2(r1.shape[0])))
    sh = (2**start, 2**(end-start), 2**(n-end))*2
    t1 = np.einsum('ijkilk->jl', r1.reshape(sh))
    t2 = np.einsum('ijkilk->jl', r2.reshape(sh))
    return la.norm(t1-t2)

def compare_all(r1, r2):
    n = int(np.round(np.log2(r1.shape[0])))
    res = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            res[i, j] = compare_segm(r1, r2, i, j+1)
    return res

def compare(rho1, rho2):
    '''Compare density matrix by trace out different degree
    For n sites, it may trace out 0, 1,..., n-1 sites
    '''
    n = int(np.round(np.log2(rho1.shape[0])))
    diff_rho = np.empty(2 * n - 1)
    diff_rho[n - 1] = la.norm(rho1 - rho2)
    for i in range(1, n):
        sh = (2**i, 2**(n - i)) * 2
        t1, t2 = rho1.reshape(sh), rho2.reshape(sh)
        r1 = np.trace(t1, axis1=1, axis2=3)
        r2 = np.trace(t2, axis1=1, axis2=3)
        diff_rho[i - 1] = la.norm(r1 - r2)
        r1 = np.trace(t1, axis1=0, axis2=2)
        r2 = np.trace(t2, axis1=0, axis2=2)
        diff_rho[n + i - 1] = la.norm(r1 - r2)
    return diff_rho
