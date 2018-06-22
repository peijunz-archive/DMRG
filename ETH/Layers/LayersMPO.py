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
O = zeros([2,2])

def MPO_H(J, g, h):
    '''$H=-\sum J Z_iZ_{i+1}+\sum (gX_i+hZ_i)$'''
    L = g*X+h*Z # local terms
    mpo = np.array([
        [I,   O,  O],
        [Z,   O,  O],
        [L, -J*Z, I],
        ])
    return mpo.transpose([1,0,2,3])

def MPO_H2(J, g, h):
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
        ])
    return mpo.transpose([1,0,2,3])

class LayersMPO(LayersStruct):
    def __init__(self, rho, H, D, W, H2):
        '''Rho is '''
        self.rho = rho
        self.H2 = H2
        self.H = H
        # As we rotated our orientation, we use D as L, L as D
        W, D = D, W
        super().__init__(W, D)
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
        delta = np.eye(2).flatten()
        deltas = reduce(np.kron, (delta,)*k)
        if n_mid == 1:
            mid = np.zeros(mpo.shape[0])
            mid[0] = 1
        else:
            mid = mpo[0].transpose([1, 0, 2]).flatten()#TODO 0,2 or 2,0
        op = np.kron(np.kron(deltas, mid), deltas)
        if n_rho:
            op = np.kron(op, rho[row]).transpose([1,2,0]).flatten()
        assert op.size == 2**(N-1)*MPO.shape[0]

    def apply_single(self, ind, op, hc=False):
        U = self[ind]
        if hc:
            U = U.T.conj()
        if ind[1] == 0 or ind[1] == self.l:
            pass

if __name__ == "__main__":
    LayersMPO((np.eye(2),)*6, 
