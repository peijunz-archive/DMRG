'''Transverse Ising Model

Tools to generate nearest interaction hamiltonian.
An important test case for the generation of H and psi

'''
import numpy as np
import scipy.sparse as sps
from functools import reduce
from spin import sigma


def nearest(n, *ops, coef=1, sparse=False):
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
    '''H=-J*Z_i x Z_{i+1}-g*X_i'''
    assert(n >= 2)
    A = (-J) * nearest(n, sigma[3], sigma[3])
    A -= g * nearest(n, sigma[1])
    return A


def Hamilton_XX(n, delta, g):
    '''$H=-\sum (Z_iZ_j-\Delta X_iX_j)-g\sum X_i$'''
    H=-nearest(n, sigma[1], coef=g)
    H-=nearest(n, sigma[3], sigma[3])
    H-=delta*nearest(n, sigma[1], sigma[1])
    return H


if __name__ == '__main__':
    import scipy.linalg as la
    A = Hamilton_trans(3, 0.1)
    print(A)
    print(*la.eigh(A), sep='\n')
