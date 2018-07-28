'''iDMRG based on MPO and MPS
NOT IMPORTANT

Use MPO to solve ground state energy
'''

import numpy as np
import scipy.sparse.linalg as SL
import scipy.linalg as LA
from .MPO import MPO


def halfshape(A):
    return A.shape[:4]


def halve(psi, sh, trunc=10):
    '''SVD psi and reshape'''
    lsh = sh[:2]
    rsh = sh[2:]
    lsize = lsh[0] * lsh[1]
    rsize = rsh[0] * rsh[1]
    A = psi.reshape([lsize, rsize])
    # Is A sparse?
    U, S, V = LA.svd(A, full_matrices=False)
    if S.size > trunc:
        d = trunc
        U = U[:, :d]
        V = V[:d]
    else:
        d = S.size
    U = U.reshape([*lsh, d])
    print(U.shape)
    V = V.reshape([d, *rsh])
    return U, S, V


def Hamiltonian(L, M, R):
    '''MPO Hamiltonian'''
    return np.einsum('Aai, ijBb, jkCc, Ddk->ABCDabcd', L, M, M, R)


def matrixify(A):
    '''Conv tensor into matrix for svd'''
    n = np.rint(np.sqrt(A.size)).astype(int)
    return A.reshape([n, n])


def next_LR(L, R, M):
    '''
    L的三个指标分别是上下中
    U分别是左下右，V也是左下右
    M分别是左右上下
    L上对U左，L下对U*左'''
    A = Hamiltonian(L, M, R)
    B = A.flatten()
    val, vec = SL.eigsh(matrixify(A), k=1, which='SA')
    U, S, V = halve(vec, halfshape(A))
    L = np.einsum('ijk, ilA, jmB, kClm', L, U, U.conj(), M)
    R = np.einsum('ijk, Ali, Bmj, Cklm', R, V, V.conj(), M)
    return L, R, val[0]


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Transverse Field Ising model H = - G * X_i + J * Z_i x Z_{i+1}
    # --------------------------------------------------------------

    M = (- MPO('sx') - MPO('sz')**2).tomatrix()
    L = np.array([[[0, 0, 1]]])
    R = np.array([[[1, 0, 0]]])

    """The chain length increase by two in every iteration"""
    for i in range(40):
        L, R, val = next_LR(L, R, M)
        print('Energy {} is'.format(2 * i + 2), val / (i * 2 + 2))
