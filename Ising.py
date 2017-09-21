'''Transverse Ising Model

Tools to generate nearest interaction hamiltonian.
An important test case for the generation of H and psi

'''
import numpy as np
import scipy.sparse as sp
from functools import reduce
from spin import sigma


def sparsity(A):
    B = A.todense()
    return np.sum(B != 0) / B.size


def nearest(n, *ops, coef=1):
    eye_n = np.eye(*ops[0].shape)
    coef *= np.ones(n)

    def f(k):
        l = [eye_n for i in range(n)]
        for i, op in enumerate(ops):
            l[k+i] = op
        return reduce(sp.kron, l)
    return sum(coef[k]*f(k) for k in range(n+1-len(ops)))

def Hamilton_trans(n, g=0, J=1):
    '''H=-J*Z_i x Z_{i+1}-g*X_i'''
    assert(n >= 2)
    A = (-J) * nearest(n, sigma[3], sigma[3])
    A -= g * nearest(n, sigma[1])
    return A


if __name__ == '__main__':
    from scipy.sparse.linalg import eigsh
    s = np.array([sparsity(Hamilton_trans(i, 0.1)) for i in range(3, 11)])
    print(s)
    A = Hamilton_trans(10, 0.1)
    print(eigsh(A, which='SA', k=1))
