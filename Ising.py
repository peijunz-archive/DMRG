'''Transverse Ising Model

An important test case for the generation of H and psi

'''
import scipy.sparse as sp
import scipy.sparse.linalg as sa
from functools import reduce
from pauli import sigma


def sparsity(A):
    B = A.todense()
    return sum(B != 0) / B.size


def Hamilton_trans(n, g=0, J=1):
    assert(n >= 2)
    A = (-J) * sum(
        reduce(sp.kron,
               (sigma[3 * (k == i or k + 1 == i)]
                for i in range(n)))
        for k in range(n - 1)
    ).astype(float64)
    A -= g * sum(
        reduce(sp.kron,
               (sigma[k == i]
                for i in range(n)))
        for k in range(n)
    )
    return A


if __name__ == '__main__':
    x = arange(3, 11)
    s = array([sparsity(Hamilton_trans(i, 0.1)) for i in x])
    print(s)
    A = Hamilton_trans(10, 0.1)
    print(sa.eigsh(A, which='SA', k=1))
