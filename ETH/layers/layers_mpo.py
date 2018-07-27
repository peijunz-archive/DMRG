from .layers import Layers
import numpy as np
from functools import reduce


def ud2rl(U):
    '''
    Up-down convention to right-left convention
                        0 1    2 0
    Convert convention  2 3 to 3 1
    '''
    return U.reshape((2,)*4).transpose([1, 3, 0, 2]).reshape(4, 4)


def rl2ud(U):
    '''Inverse of ud2rl
    Right-left convention to up-down convention
                        2 0    0 1
    Convert convention  3 1 to 2 3
    '''
    return U.reshape((2,)*4).transpose([2, 0, 3, 1]).reshape(4, 4)


from DMRG.spin import sigma
# Pauli Matrices
I, X, Y, Z = sigma
# Zero Matrix
O = np.zeros([2, 2])


def MPO_TL(J=1, g=1, h=1):
    '''$H=-\sum J Z_iZ_{i+1}+\sum (gX_i+hZ_i)$
    NOTE The matrix of operators is upper-triangle after transosition
    '''
    L = g*X+h*Z  # local terms
    LZ = (L@Z+Z@L)/2
    mpo = np.array([
        [I,     O,      O,      O,      O,      O],
        [Z,     O,      O,      O,      O,      O],
        [L,    -J*Z,    I,      O,      O,      O],
        [Z@Z,   O,      O,      O,      O,      O],
        [LZ,   -J*Z@Z,  Z,      O,      O,      O],
        [L@L,  -2*J*LZ, 2*L,  J*J*Z@Z,  -2*J*Z,  I],
    ]).transpose([1, 0, 2, 3])
    return mpo[:3, :3], mpo


def transform(op, U, i):
    '''i is the index of slot. If i>0 the starting shape is 2**i, if i<0, op.size*s**i
       <-i->U
    op ------------

    '''
    if i >= 0:
        start = 2**i
    else:
        start = op.size//2**(-i)
    tail = op.size//start
    if tail < U.shape[0]:
        # wrap index to the back
        s2 = U.shape[0]//tail
        op = op.reshape([s2, -1, tail])
        U = U.reshape([tail, s2, tail, s2])
        return np.einsum('ijkl, lmk->jmi', U, op)
    else:
        op = op.reshape([start, U.shape[0], -1])
        return np.einsum('ij,kjl->kil', U, op)


def dagger(U):
    '''
    i j   ---\  j i *
    k l   ---/  l k
    '''
    U = U.reshape((2,)*4).transpose([1, 0, 3, 2])
    return U.reshape(4, 4).conj()


class LayersMPO(Layers):
    def __init__(self, rho, H, D, W, H2, offset=0):
        '''Rho is '''
        self.rho = rho
        self.H2 = H2
        self.H = H
        # As we rotated our orientation, we use D as L, L as D
        W, D = D, W
        super().__init__(W, D, offset)
        I = ud2rl(np.eye(4))
        self.U[:] = I[np.newaxis, np.newaxis]

    def init_operator(self, mpo, row=0):
        '''Number of indices is 2*(# of width)-1
        1 on the edge，is dangling 
        3 in the middle is matrix operator
        (1,2,2,1,2,2,1)
        (2,2,3,2,2)
        (1,2,2,3,2,2,1)
        (2,2,2,1,2,2,2)
        上三角MPO
        on demand consumption of rho and MPO
        '''
        N_legs = 2*(self.W-1)+1
        # n_delta is number of deltas in one side
        # n_mid is number of legs binded to MPO
        # print(())
        assert row in [0, -1], "Invalid init operator in the middle"
        space = not((row, 0) in self)
        n_delta, n_mid = divmod(N_legs-2*space, 4)
        s = [*("δ")*n_delta, str(n_mid), *("δ")*n_delta]
        if space:
            s = ["ρ"]+s+["ρ"]
        print("{} Legs, {} gap:\t".format(N_legs, int(space)), *s)
        deltas = reduce(np.kron, (1,)+(np.eye(2).flatten(),)*n_delta)
        if n_mid == 1:
            mid = np.zeros(mpo.shape[0])
            mid[row] = 1
        else:
            mpo_ = mpo[0] if row == 0 else mpo[:, -1]
            mid = mpo_.transpose([1, 0, 2]).flatten()  # TODO 1,2 or 2,1
        op = np.kron(np.kron(deltas, mid), deltas)
        if space:
            op = np.einsum('i, jk->kij', op, self.rho[row]).flatten()
        size = 2**(self.W-1)
        return op.reshape(size, mpo.shape[0], size)

    def apply_pair(self, ind, op, mpo, rhs=False):
        R = self[ind]
        if self.W-1 == 0:
            '''Contract rho and MPO at the same time'''
            raise NotImplementedError
        elif ind[1] == 0:
            #print("contract rho")
            '''Contract rho and then apply'''
            # contract dagger(U) rho U, and apply to -ind[1]-1
            U = rl2ud(R)
            rho = np.kron(self.rho[ind[0]], self.rho[ind[0]+1])
            combo = ud2rl(U.T.conj()@rho@U)
            if rhs:
                combo = combo.T
            return transform(op, combo, -ind[1]-1)
        elif ind[1] == self.W-1:
            '''Contract MPO and then apply'''
            # Contract U MPO dagger(U) and then apply
            U = rl2ud(R)
            # Double MPO may be cached
            #   k       m         k m
            # i + j * j + o --> i +++ o
            #   l       n         l n
            dmpo = np.einsum('ijkl, jomn->iokmln', mpo, mpo)
            dmpo = dmpo.reshape(*mpo.shape[:2], *U.shape)
            # Apply U to upper indices and lower indices
            combo = np.einsum('ij, mnjk, kl->mnil', U, dmpo, U.T.conj())
            combo = combo.reshape(*mpo.shape[:2], *(2,)*4)
            #   2 3           3 0
            # 0 +++ 1  -->  4 +++ 1
            #   4 5           5 2
            combo = combo.transpose(
                [3, 1, 5, 2, 0, 4]).reshape(mpo.shape[0]*4, -1)
            if rhs:
                combo = combo.T
            return transform(op, combo, ind[1]-1)
        else:
            '''Simply apply'''
            if rhs:
                R = R.T
            op = transform(op, R, ind[1]-1)
            op = transform(op, dagger(R), -ind[1]-1)
            return op

    def contract_hole(self, ind, l, r, mpo):
        R = self[ind]
        if self.W-1 == 0:
            raise NotImplementedError
        elif ind[1] == 0:
            #print("contract rho")
            '''Contract rho and then apply'''
            # contract dagger(U) rho U, and apply to -ind[1]-1
            l = l.reshape(2, -1, 2)
            r = r.reshape(2, -1, 2)
            lr_holes = np.einsum('imk, jml->ijkl', l, r)
            return np.einsum('ijkl, ab, cd->bdijackl').reshape([4]*4)
        elif ind[1] == self.W-1:
            dmpo = np.einsum('ijkl, jomn->iokmln', mpo, mpo)
            dmpo = dmpo.reshape(*mpo.shape[:2], *U.shape)
            c = int(sqrt(dmpo.size))
            lr = int(sqrt(l.size/c))
            l = l.reshape([lr, 2, -1, 2, lr])
            r = r.reshape([lr, 2, -1, 2, lr])
            hole = np.einsum('ijklm,inopm,koqrst->jqnrlspt', l, r, dmpo)  # ???
            return hole
        else:
            l = l.reshape([lr, 2, -1, 2, lr])

    def contract_list(self, inds, op, mpo, rhs=False):
        if rhs:
            for i in inds[::-1]:
                op = self.apply_pair(i, op, mpo, True)
        else:
            # print(op.flatten())
            for i in inds:
                op = self.apply_pair(i, op, mpo)
                #print(i, op.flatten())
        return op

    def contract_all(self, mpo, rhs=False):
        "Test that contract all elements in different ways will give correct energy"
        Left = self.init_operator(mpo, row=0)
        Right = self.init_operator(mpo, row=-1)
        if rhs:
            Right = self.contract_list(self.indices, Right, mpo, True)
        else:
            Left = self.contract_list(self.indices, Left, mpo)
        return np.dot(Left.flatten(), Right.flatten())

    def sweep(self, mpo, N):
        L = [self.init_operator(mpo, row=0)]
        R = [self.init_operator(mpo, row=-1)]
        for i in self.indices[1:][::-1]:
            R.append(self.apply_pair(i, R[-1], mpo, True))
        nblocks = len(self.indices)
        for i in range(N):
            # Forward
            for j in self.indices[0:nblocks-1]:
                yield j, L[-1], R[-1]
                L.append(self.apply_pair(j, L[-1], mpo))
                R.pop()
            # Backward
            for j in self.indices[nblocks-1:0:-1]:
                yield j, L[-1], R[-1]
                L.pop()
                R.append(self.apply_pair(j, R[-1], mpo, True))


if __name__ == "__main__":
    from ETH import Rho
    from numpy.random import RandomState
    l = 4
    # [np.array([[0.7,0],[0,0.3]]) for i in range(l)]
    rho = Rho.rho_even(l, l/2, amp=0.4, rs=RandomState(123581321))
    H, H2 = MPO_TL(J=1, g=1, h=1)
    L = LayersMPO(rho, H, 2, l-1, H2, offset=0)
    Left = L.init_operator(H, row=0)
    #print(L.apply_pair((0,0), Left, H).flatten())
    print("Final", L.contract_all(L.H).real)
    h = [np.einsum('ijkl, lk->ij', H, r) for r in rho]
    print(reduce(np.matmul, h)[0, -1].real)
    print("MPO 1", np.einsum("i, jk->kij", h[0][0], rho[1]).flatten())
    #print(np.einsum("ijk, kj, lm->mil", H[0], rho[0], rho[1]).flatten())
    #print("MPO 2", np.einsum("i, iklm->klm", (h[0]@h[1])[0], H).transpose([1,0,2]).flatten())
    #print("MPO 3", np.einsum("i, ikll, mn->nkm", (h[0]@h[1])[0], H, rho[0]).flatten())
