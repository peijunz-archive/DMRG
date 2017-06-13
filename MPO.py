from numpy import zeros, array, einsum, tensordot, matmul
import math
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh, norm, svd
from scipy.sparse.linalg import svds

G = 0.1
J = 1.0
D = 10
s0 = array([[1, 0], [0, 1]], dtype='int8')
sx = array([[0, 1], [1, 0]], dtype='int8')
sz = array([[1, 0], [0, -1]], dtype='int8')

M = zeros([3, 3, 2, 2], dtype='double')
M[0, 0] = s0
M[2, 2] = s0
M[2, 0] = -G * sx
M[2, 1] = sz
M[1, 0] = -J * sz

S = 2

L = zeros([1, 1, 3])
R = zeros([1, 1, 3])

L[:, :, -1] = 1
R[:, :, 0] = 1

#


def isqrt(n):
    return round(math.sqrt(n))


def halve(psi, sh):
    '''SVD psi'''
    lsh = sh[:2]
    rsh = sh[2:]
    lsize = lsh[0] * lsh[1]
    rsize = rsh[0] * rsh[1]
    A = psi.reshape([lsize, rsize])
    U, S, V = svd(A, full_matrices=False)
    if S.size > D:
        d = D
        U = U[:, :D]
        V = V[:D]
    else:
        d = S.size
    U = U.reshape([*lsh, d])
    V = V.reshape([d, *rsh])
    print(U.size)
    return U, S, V


def H(L, M, R):
    return einsum('Aa i, ij Bb, jk Cc,Dd k->ABCDabcd', L, M, M, R)


def halfshape(A):
    return A.shape[:4]


def matrixify(A):
    n = isqrt(A.size)
    return A.reshape([n, n])


def next_LR(L, R, M):
    '''L的三个指标分别是上下中
    U分别是左下右，V也是左下右
    M分别是左右上下
    L上对U左，L下对U*左'''
    A = H(L, M, R)
    val, vec = eigsh(matrixify(A), k=1, which='SA')
    U, S, V = halve(vec, halfshape(A))
    L = einsum('ijk, ilA, jmB, kClm', L, U, U.conj(), M)
    R = einsum('ijk, Ali, Bmj, Cklm', R, V, V.conj(), M)
    return L, R, val


for i in range(20):
    L, R, val = next_LR(L, R, M)
    print('Energy is', val)
