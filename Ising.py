'''Ising Model Hamiltonian Template Generator

Tools to generate n-nearest interaction Hamiltonian.

Transverse field Ising model is an important test case for
the generation of H and psi

'''
import numpy as np
import scipy.sparse as sps
from functools import reduce
from spin import sigma


def nearest(n, *ops, coef=1, sparse=False):
    '''Generate k nearest Hamiltonian term
    $$\sum_{i=0}^{n-k} \bigotimes_{j=0}^{k-1} \Omega^k_{i+j}$$
    , k is number of operators

    Args:
    n       length of chain
    ops     Operator list
    coef    Coefficients of term at some site
    Examples:
    n sites $\sum_i X_i$ can be generated by nearest(6, X)
    n sites $\sum_i X_iY_{i+1}$ can be generated by nearest(6, X, Y)
    '''
    eye_n = np.eye(*ops[0].shape)
    coef *= np.ones(n)
    def _local_H(k):
        l = [eye_n for i in range(n)]
        for i, op in enumerate(ops):
            l[k+i] = op
        if sparse:
            return reduce(sps.kron, l)
        else:
            return reduce(np.kron, l)
    return sum(coef[k]*_local_H(k) for k in range(n+1-len(ops)))

def Hamilton_trans(n, g=0, J=1):
    '''$H=-J*Z_i x Z_{i+1}-g*X_i$'''
    assert(n >= 2)
    A = (-J) * nearest(n, sigma[3], sigma[3])
    A -= g * nearest(n, sigma[1])
    return A


def Hamilton_XX(n=None, delta=1/2, g=1, rs=None):
    '''$H=-\sum (Z_iZ_j+\Delta X_iX_j)-\sum g_iX_i$'''
    if rs is not None:
        g = rs.uniform(-g, g, n)
    if n is None:
        return Hamilton_XZ.__doc__
    H=np.zeros([2**n, 2**n], dtype='complex128')
    H-=nearest(n, sigma[1], coef=g)
    H-=nearest(n, sigma[3], sigma[3])
    H-=delta*nearest(n, sigma[1], sigma[1])
    return H

def Hamilton_XZ(n=None, delta=1/2, g=1, rs=None, h=0.1):
    '''$H=-\sum (X_iX_j+Y_iY_j-\Delta Z_iZ_j)+\sum gX_i+hZ_i$'''
    if rs is not None:
        g = rs.uniform(-g, g, n)
    if n is None:
        return Hamilton_XZ.__doc__
    H=np.zeros([2**n, 2**n], dtype='complex128')
    H-=nearest(n, sigma[1], sigma[1])
    H-=nearest(n, sigma[2], sigma[2])
    H-=delta*nearest(n, sigma[3], sigma[3])
    H+=nearest(n, sigma[1], coef=g)
    H+=nearest(n, sigma[3], coef=h)
    return H

def Hamilton_TL(n=None, J=1, g=0.945, rs=None, h=0.8090):
    '''$H=-\sum J Z_iZ_j+\sum gX_i+hZ_i$'''
    if n is None:
        return Hamilton_TL.__doc__
    print(J, g, h)
    H=np.zeros([2**n, 2**n], dtype='complex128')
    H+=J*nearest(n, sigma[3], sigma[3])
    H+=g*nearest(n, sigma[1])
    H+=h*nearest(n, sigma[3])
    return H

if __name__ == '__main__':
    import scipy.linalg as la
    from numpy.random import RandomState

    A = Hamilton_XX(4, 0.5, 1, rs=RandomState(5))
    print(A)
    #print(*la.eigh(A), sep='\n')
