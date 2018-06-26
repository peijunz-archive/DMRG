from .Layers import LayersStruct
import numpy as np
from functools import reduce

def rotl(U):
    return U.reshape((2,)*4).transpose([2, 0, 3, 1]).reshape(4, 4)

def rotr(U):
    return U.reshape((2,)*4).transpose([1, 3, 0, 2]).reshape(4, 4)

def index_structure(N, gapless):
    if gapless:
        return (0, *divmod(N, 4))
    else:
        return (1, *divmod(N-2, 4))

from DMRG.spin import sigma
# Pauli Matrices
I, X, Y, Z = sigma
# Zero Matrix
O = np.zeros([2,2])

def MPOs(J=1, g=1, h=1):
    '''$H=-\sum J Z_iZ_{i+1}+\sum (gX_i+hZ_i)$'''
    L = g*X+h*Z # local terms
    LZ = (L@Z+Z@L)/2
    mpo = np.array([
        [I,     O,      O,      O,      O,      O],
        [Z,     O,      O,      O,      O,      O],
        [L,    -J*Z,    I,      O,      O,      O],
        [Z@Z,   O,      O,      O,      O,      O],
        [LZ,   -J*Z@Z,  Z,      O,      O,      O],
        [L@L,  -2*J*LZ, 2*L,  J*J*Z@Z,  2*J*Z,  I],
        ]).transpose([1,0,2,3])
    return mpo[:3, :3], mpo

def transform(op, U, i):
    '''i is the index of slot. If i>0 the starting shape is 2**i, if i<0, op.size*s**i'''
    if i>= 0:
        start = 2**i
    else:
        start = op.size//2**(-i)
    s1 = op.size//start
    if s1 < U.shape[0]:
        # wrap index to the back
        s2 = U.shape[0]//s1
        op = op.reshape([s2, s1, s1])
        U = U.reshape([s1, s2, s1, s2])
        return np.einsum('ijkl, lmk->jmi', U, op)
    else:
        op = op.reshape([start, U.shape[0], -1])
        return np.einsum('ij,kjl->kil', U, op)

def dagger(U):
    '''
    i j   ---\  j i *
    k l   ---/  l k
    '''
    U = U.reshape((2,)*4).transpose([1,0,3,2])
    return U.reshape(4, 4).conj()

class LayersMPO(LayersStruct):
    def __init__(self, rho, H, D, W, H2, offset=0):
        '''Rho is '''
        self.rho = rho
        self.H2 = H2
        self.H = H
        # As we rotated our orientation, we use D as L, L as D
        W, D = D, W
        super().__init__(W, D, offset)
        I = rotl(np.eye(4))
        self.U[:] = I[np.newaxis, np.newaxis]

    def init_operator(self, mpo, row=0):
        '''Number of indices is 2*(# of width)-1
        1之存在于边缘，乃边缘空缺也
        3之存在于间，乃矩阵乘积算子也
        (1,2,2,1,2,2,1)
        (2,2,3,2,2)
        (1,2,2,3,2,2,1)
        (2,2,2,1,2,2,2)
        上三角MPO
        '''
        N = 2*self.W-1
        gapless = (row, 0) in self
        n_rho, n_delta, n_mid = index_structure(N, gapless)
        print(n_rho, n_delta, n_mid)
        delta = np.eye(2).flatten()
        deltas = reduce(np.kron, (delta,)*n_delta)
        if n_mid == 1:
            mid = np.zeros(mpo.shape[0])
            mid[0] = 1
        else:
            mid = mpo[0].transpose([1, 0, 2]).flatten()#TODO 0,2 or 2,0
        op = np.kron(np.kron(deltas, mid), deltas)
        if n_rho:
            op = np.einsum('i, jk->kij', op, rho[row]).flatten()
        size = 2**(self.W-1)
        return op.reshape(size, mpo.shape[0], size)

    def apply_pair(self, ind, op, hc=False):
        U = self[ind]
        if self.W == 1:
            '''Contract rho and MPO at the same time'''
            return op
        elif ind[1] == 0:
            '''Contract rho and then apply'''
            # contract dagger(U) rho U, and apply to -ind[1]-1
            U = rotr(U)
            rho = kron(self.rho[ind[0]], self.rho[ind[0]+1])
            combo = rotl(U.T.conj()@rho@U)
            return transform(op, combo, -ind[1]-1)
        elif ind[1] == self.L-1:
            '''Contract MPO and then apply'''
            # Contract U MPO dagger(U) and then apply
            U = rotr(U)
            # Dual MPO Can be cached
            dmpo = np.einsum('ijkl, jomn->iojmkn', MPO, MPO)
            dmpo = dmpo.reshape(*MPO.shape[:2], 2,2)
            combo = np.einsum('ij, mnjk, kl->mnil', U, dmpo, U.T.conj())
            combo = combo.reshape(*MPO.shape[:2], *(2,)*4)
            # mnijlk->ilnjkm
            combo = combo.transpose([2, 4, 0, 3, 5, 1]).reshape(MPO.shape[0]*4, -1)
            return transform(op, combo, ind[1]-1)
        else:
            '''Simply apply'''
            op = transform(op, U, ind[1]-1)
            return transform(op, dagger(U), -ind[1]-1)

if __name__ == "__main__":
    l = 6
    rho = (np.eye(2),)*l
    H, H2 = MPOs()
    L = LayersMPO(rho, H, 4, l-1, H2, offset=1)
    print(L.indices)
    print(L.init_operator(L.H).shape)
