'''Matrix product state class

Infrastructure for DMRG and iTEBD

'''
import numpy as np
import numpy.linalg as la
from functools import reduce


class State:
    '''Length:
    + o: L-1
    + v: L
    + chi: L+1
    '''

    def __init__(self, arg=(1, 1), dim=2):
        '''
        arg may be chi or M, only one of them is needed.
        chi is an array of truncation length with length n+1,
        so we have chi[0]...chi[n] and has shape (chi[i], chi[i+1])
        for M[i] where i is in range(0, n).

        Basically, if we are discussing MPS, chi[0]=chi[n]=1. But if we keep
        some boundary open, for example chi[n] > 1, then they is a series of MPS,
        which may be used to construct some other MPS on contraction.
        '''
        self.dim = dim
        if isinstance(arg[0], np.ndarray):
            self.from_M(arg)
        else:
            self.from_chi(arg)
        self.lamb = [None] * (self.L - 1)
        self.canonical = False

    def from_chi(self, chi):
        self.chi = list(chi)
        self.M = [np.zeros([l, self.dim, r], 'complex')
                  for l, r in zip(chi[:-1], chi[1:])]
        self.L = len(chi) - 1

    def from_M(self, M):
        L = [m.shape[0] for m in M]
        R = [m.shape[0] for m in M]
        assert L[1:] == R[:-1], "Incompatible matrices!"
        L.append(R[-1])
        self.chi = L
        self.M = [np.array(i, 'complex') for i in M]
        self.L = len(M)

    @staticmethod
    def naive(*L):
        '''L is N*s matrix, chi are ones'''
        L = np.array(L).reshape([-1, 1, 2, 1])
        return State(L)

    def graph(self):
        '''
        1 -- v -- 2 -- v -- 1
             |         |
             2         2
        '''
        s = '-- v --'.join('{:^3}'.format(i) for i in self.chi)
        w = ''.join(['|' if i == 'v' else ' ' for i in s])
        n = w.replace('  |  ', '{:^5}'.format(self.dim))
        return '\n'.join((s, w, n))

    def __repr__(self):
        return self.graph()

    def __str__(self):
        return self.graph()

    def shape(self, i):
        return self.chi[i], self.dim, self.chi[i + 1]

    def ortho_left_site(self, i, cut=True, err=1e-16):
        '''Left orthogonalize site i and optimize left chi[i+1]'''
        before = (-1, self.chi[i + 1])
        after = (self.chi[i], self.dim, -1)
        m = self.M[i].reshape(before)
        u, s, v = la.svd(m, full_matrices=False)
        if cut:
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i + 1] = ind
        self.M[i] = u.reshape(after)
        return s, v

    def ortho_right_site(self, i, cut=True, err=1e-16):
        '''Right orthogonalize site i and optimize left chi[i]'''
        before = (self.chi[i], -1)
        after = (-1, self.dim, self.chi[i + 1])
        m = self.M[i].reshape(before)
        u, s, v = la.svd(m, full_matrices=False)
        if cut:
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i] = ind
        self.M[i] = v.reshape(after)
        print(self.M[i].dtype, v.dtype)
        print('After M = {}, v = {}'.format(self.M[i], v.reshape(after)))
        return u, s

    def ortho_left(self, cen=None):
        '''Go from the left end to left orthogonalize till center'''
        if cen is None:
            cen = self.L - 1
        for i in range(cen):
            s, v = self.ortho_left_site(i)
            s /= la.norm(s)
            self.lamb[i] = s
            v *= s[:, np.newaxis]
            self.M[i + 1] = np.einsum('al, lcr->acr', v, self.M[i + 1])
        self.verify_shape()

    def ortho_right(self, cen=0):
        '''Go from the right end to right orthogonalize till center'''
        for i in range(self.L - 1, cen, -1):
            u, s = self.ortho_right_site(i)
            print('-' * 30)
            s /= la.norm(s)
            self.lamb[i - 1] = s
            u *= s[np.newaxis, :]
            self.M[i - 1] = np.einsum('lcr, ra->lca', self.M[i - 1], u)  # TODO
        self.verify_shape()

    def canon(self):
        '''Decompose MPS into ΓΛΓΛΓΛΓΛΓ'''
        self.ortho_left()
        self.ortho_right()
        for i in range(0, self.L - 1):
            print('Shape of M', self.M[i].shape)
            self.M[i] /= self.lamb[i][np.newaxis, np.newaxis, :]
        self.canonical = True

    def A(self, i):
        '''Left Matrix after Orthonormalization'''
        if i == 0:
            return self.M[0]
        else:
            return self.M[i] * self.lamb[i - 1][:, np.newaxis, np.newaxis]

    def B(self, i):
        '''Left Matrix after Orthonormalization'''
        if i == self.L - 1:
            return self.M[-1]
        else:
            return self.M[i] * self.lamb[i][np.newaxis, np.newaxis, :]

    def verify_shape(self):
        '''Verify that chi is compatible with M'''
        for i in range(self.L):
            expect = self.shape(i)
            got = self.M[i].shape
            assert expect == got, 'Shape error at {}, expected {}, got {}'.format(
                i, expect, got)

    def dot(self, rhs):
        '''Inner product between two wavefunctions'''
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, lcj->rj', self.S(0).conj(), rhs.S(0))
        for i in range(1, self.L):
            E = np.einsum('kl, kno, lnr->or', E, self.S(i).conj(), rhs.S(i))
        return np.trace(E)

    def wave(self):
        '''Get the wave function'''
        pass

    def __getitem__(self, ind):
        '''Bloch Wave function for each site'''
        return State.naive(ind).dot(self)

    def __add__(self, m):
        l, d, r = m.shape
        if d == self.dim and l == self.chi[-1]:
            self.M.append(m)
            self.chi.append(l2)

    def measure(self, *ops):
        '''oplist should be a list of ordered operators like
        (4, sigma[3]), (6, sigma[2])
        '''
        assert(self.canonical)
        x = ops[0][0]
        E = np.eye(self.chi[x], dtype='complex')
        for i, op in ops:
            if x < i:
                for j in range(x, j):
                    E = np.einsum('kl, kio, lir->or',
                                  E, self.A(i).conj(), self.A(i))
            x = i + 1
            E = np.einsum('kl, kio, ij, ljr->or',
                          E, self.A(i).conj(), op, self.A(i))
        return np.einsum('ii, i', E, self.lamb[i]**2).real

    def update(U, *sites):
        '''Apply U at sites, there are 2*len(sites) axes in U'''
        assert(len(sites) * 2 == len(U.shape))
        if sites:
            pass


if __name__ == "__main__":
    from pauli import sigma
    #s = State((1, 2, 1))
    #s.M[0][0, :, :] = np.array([[5, 2 + 1], [1, 6j]])
    #s.M[1][:, :, 0] = np.array([[5 - 1j, 2 - 3j], [0.3, 4]])
    # np.set_printoptions(precision=5)
    # s.canon()
    # s.dot(s)
    r = State.naive([0, 1], [1, 0], [2, 3])
    r.canon()
    print(r.lamb)
