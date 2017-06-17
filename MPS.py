'''Matrix product state class

Infrastructure for DMRG and iTEBD

'''
import numpy as np
import numpy.linalg as la
from functools import reduce


class State:
    '''Length:
    + o: L+1
    + chi: L+1
    + v: L+2
    '''

    def __init__(self, chi=None, M=None, dim=2):
        '''chi is an array of truncation length with length n+1,
        so we have chi[0]...chi[n] and has shape (chi[i], chi[i+1])
        for M[i] where i is in range(0, n).

        Basically, if we are discussing MPS, chi[0]=chi[n]=1. But if we keep
        some boundary open, for example chi[n] > 1, then they is a series of MPS,
        which may be used to construct some other MPS on contraction.
        '''
        chi = list(chi)
        self.L = len(chi) - 1
        self.chi = chi
        self.M = [None] * (self.L + 2)
        self.M[0] = np.ones([1, 1, 1], 'complex')
        self.M[-1] = np.ones([1, 1, 1], 'complex')
        self.M[1:-1] = [np.zeros([l, dim, r], 'complex')
                        for l, r in zip(chi[:-1], chi[1:])]
        self.dim = dim
        self.status = 'raw'
        self.lamb = [None] * (self.L + 1)
        self.lamb[0] = np.ones([1])
        self.lamb[-1] = np.ones([1])
        self.canonical = False

    def from_chi(self):
        pass

    def from_M(self):
        pass

    @staticmethod
    def naive(L):
        '''L is N*s matrix, chi are ones'''
        n = len(L)
        chi = [1] * (n + 1)
        s = State(chi)
        L = np.array(L)
        for i in range(n):
            s.M[i + 1] = L[i].reshape((1, 2, 1))
        return s

    def graph(self):
        '''
            1--▼--2--▼--1
               |     |
               2     2
            1--v--2--v--1
        '''
        s = '--▼--'.join(str(i) for i in self.chi)
        w = ''.join(['|' if i == '▼' else ' ' for i in s])
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
        m = self.M[i + 1].reshape(before)
        u, s, v = la.svd(m, full_matrices=False)
        if cut:
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i + 1] = ind
        self.M[i + 1] = u.reshape(after)
        return s, v

    def ortho_right_site(self, i, cut=True, err=1e-16):
        '''Right orthogonalize site i and optimize left chi[i]'''
        before = (self.chi[i], -1)
        after = (-1, self.dim, self.chi[i + 1])
        m = self.M[i + 1].reshape(before)
        u, s, v = la.svd(m, full_matrices=False)
        if cut:
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i] = ind
        self.M[i + 1] = v.reshape(after)
        return u, s

    def ortho_left(self, cen=None):
        '''Go from the left end to left orthogonalize till center'''
        if cen is None:
            cen = self.L
        assert self.chi[0] == 1, "Left end should be closed!"
        for i in range(cen):
            #print('Before:', np.einsum('lcr, rst->lcst', self.M[i+1], self.M[i+2]))
            s, v = self.ortho_left_site(i)
            self.lamb[i + 1] = s
            v *= s[:, np.newaxis]
            self.M[i + 2] = np.einsum('al, lcr->acr', v, self.M[i + 2])
            #print('After:', np.einsum('lcr, rst->lcst', self.M[i+1], self.M[i+2]))
        if cen == self.L and self.lamb[-1][0] < 0:
            self.lamb[-1][0] *= -1
            self.M[self.L] *= -1
            self.M[self.L + 1] *= -1
        self.verify_shape()

    def ortho_right(self, cen=0):
        '''Go from the right end to right orthogonalize till center'''
        assert self.chi[-1] == 1, "Right end should be closed!"
        for i in range(self.L - 1, cen - 1, -1):
            u, s = self.ortho_right_site(i)
            self.lamb[i] = s
            u *= s[np.newaxis, :]
            self.M[i] = np.einsum('lcr, ra->lca', self.M[i], u)
        if cen == 0 and self.M[0] < 0:
            self.M[0] *= -1
            self.M[1] *= -1
        self.verify_shape()

    def canon(self):
        '''Decompose MPS into ΓΛΓΛΓΛΓΛΓ'''
        self.ortho_left()
        self.ortho_right()
        self.lamb[-1][0] = 1
        for i in range(0, self.L):
            self.M[i + 1] /= self.lamb[i + 1][np.newaxis, np.newaxis, :]
        assert (np.angle(self.M[0]) == 0 and np.angle(
            self.M[-1]) == 0), "Phase != 0 at ends!"
        self.canonical = True

    def S(self, i):
        '''Safe Matrix visit'''
        if self.canonical:
            return self.A(i)
        else:
            return self.M[i + 1]

    def A(self, i):
        '''Left Matrix after Orthonormalization'''
        return self.M[i + 1] * self.lamb[i][:, np.newaxis, np.newaxis]

    def B(self, i):
        '''Left Matrix after Orthonormalization'''
        return self.M[i + 1] * self.lamb[i + 1][np.newaxis, np.newaxis, :]

    def verify_shape(self):
        '''Verify that chi is compatible with M'''
        for i in range(self.L):
            expect = self.shape(i)
            got = self.M[i + 1].shape
            assert expect == got, 'Shape error at {}, expected {}, got {}'.format(
                i, expect, got)

    def dot(self, rhs):
        '''Inner product between two wavefunctions'''
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, icj->rj', self.S(0).conj(), rhs.S(0))
        for i in range(1, self.L):
            print(E)
            E = np.einsum('kl, kno, lnr->or', E, self.S(i).conj(), rhs.S(i))
        return E[0, 0]

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
        return np.einsum('ii, i', E, self.lamb[i + 1]**2).real

    def update(sites, U):
        '''Apply U at sites, there are 2*len(sites) axes in U'''
        assert(len(sites) * 2 == len(U.shape))
        assert()


if __name__ == "__main__":
    from pauli import sigma
    s = State((1, 2, 1))
    s.M[1][0, :, :] = np.array([[0, 2 + 1], [1, 0]])
    s.M[2][:, :, 0] = np.array([[5 - 1, 2 - 3j], [0.3, 4]])
    np.set_printoptions(precision=5)
    s.canon()
    print(s.lamb[1])
    #print(s.measure((0, sigma[3])))
    # print(s.lamb[1]**2)
    # print(s.dot(s))
    # print(s[[0,1],[0,1]])
    # print(s[[0,1],[1,0]])
    # print(s[[1,0],[0,1]])
    # print(s[[1,0],[1,0]])
    print(0.317588170991**2 + 0.414245440423**2 + 0.666302284103**2)
