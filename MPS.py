from numpy import empty, zeros, matmul, array
import numpy as np
import numpy.linalg as la
from numpy.linalg import eigh, norm, svd
from functools import reduce


class MPS:
    '''How to construct?
    + Given chi and construct basic shape
    + Given matrices and then construct?
    + MPS by appromimating a quantum state
    + Linear combination?
    Length:
    + o: L+1
    + chi: L+1
    + v: L+2
    '''

    def __init__(self, chi, dim=2, orth=True):
        '''chi is an array of truncation length with length n+1,
        so we have chi[0]...chi[n] and has shape (chi[i], chi[i+1])
        for M[i] where i is in range(0, n).

        Basically, if we are discussing MPS, chi[0]=chi[n]=1. But if we keep
        some boundary open, for example chi[n] > 1, then they is a series of MPS,
        which may be used to construct some other MPS on contraction.
        '''
        chi = list(chi)
        self.L = len(chi) - 1
        for i in range(self.L):
            chi[i + 1] = min(2 * chi[i], chi[i + 1])
            chi[self.L - (i + 1)] = min(2 * chi[self.L - i],
                                        chi[self.L - (i + 1)])
        self.chi = chi
        self.M = [None] * (self.L + 2)
        self.M[0] = np.ones([1, 1, 1], dtype='complex')
        self.M[-1] = np.ones([1, 1, 1], dtype='complex')
        self.M[1:-1] = [zeros([chi[i], dim, chi[i + 1]],
                              dtype='complex') for i in range(self.L)]
        self.dim = dim
        self.status = 'raw'
        self.lamb = [None] * (self.L + 1)
        self.lamb[0] = np.ones([1])
        self.lamb[-1] = np.ones([1])

    def graph(self):
        ''' 1--v--2--v--1
               |     |
               2     2
        '''
        s = '--▼--'.join(str(i) for i in self.chi)
        w = ''.join(['|' if i == '▼' else ' ' for i in s])
        n = w.replace('  |', '{:3d}'.format(self.dim))
        return '\n'.join((s, w, n))

    def shape(self, i):
        return self.chi[i], self.dim, self.chi[i + 1]

    def ortho_left_site(self, i, cut=True, err=1e-16):
        before = (-1, self.chi[i + 1])
        after = (self.chi[i], self.dim, -1)
        m = self.M[i + 1].reshape(before)
        u, s, v = svd(m, full_matrices=False)
        if cut:
            # print(s)
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i + 1] = ind
        self.M[i + 1] = u.reshape(after)
        return s, v

    def ortho_right_site(self, i, cut=True, err=1e-16):
        before = (self.chi[i], -1)
        after = (-1, self.dim, self.chi[i + 1])
        m = self.M[i + 1].reshape(before)
        u, s, v = svd(m, full_matrices=False)
        if cut:
            ind = sum(np.abs(s) > err)
            u = u[:, :ind]
            v = v[:ind]
            s = s[:ind]
            self.chi[i] = ind
        self.M[i + 1] = v.reshape(after)
        return u, s

    def ortho_left(self, cen=None):
        '''Orthonormalization from the left the the end'''
        if cen is None:
            cen = self.L
        assert(self.chi[0] == 1)
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
        assert(self.verify_shape())

    def ortho_right(self, cen=0):
        assert(self.chi[-1] == 1)
        for i in range(self.L - 1, cen - 1, -1):
            u, s = self.ortho_right_site(i)
            self.lamb[i] = s
            u *= s[np.newaxis, :]
            self.M[i] = np.einsum('lcr, ra->lca', self.M[i], u)
        if cen == 0 and self.M[0] < 0:
            self.M[0] *= -1
            self.M[1] *= -1
        assert(self.verify_shape())

    def vidalize(self):
        '''Gamma*Lambda=B'''
        self.ortho_left()
        self.ortho_right()
        self.lamb[-1][0] = 1
        for i in range(0, self.L):
            self.M[i + 1] /= self.lamb[i + 1][np.newaxis, np.newaxis, :]
        assert(np.angle(s.M[0]) == 0 and np.angle(s.M[-1]) == 0)

    def A(self, i):
        return self.M[i + 1] * self.lamb[i][:, np.newaxis, np.newaxis]

    def B(self, i):
        return self.M[i + 1] * self.lamb[i + 1][np.newaxis, np.newaxis, :]

    def verify_shape(self):
        for i in range(self.L):
            expect = self.shape(i)
            got = self.M[i + 1].shape
            if expect != got:
                print('Shape error at {}, expected {}, got {}'.format(i, expect, got))
                return False
        return True

    def __repr__(self):
        return self.graph()

    def dot(self, rhs):
        '''Inner product between two wavefunction'''
        pass

    def wave(self):
        '''Get the wave function'''
        pass

    def __getitem__(self, ind):
        print(ind)
        return reduce(matmul, (self.M[i][ind[i]] for i in range(self.dim)))[0, 0]

    def __add__(self, m):
        l, d, r = m.shape
        if d == self.dim and l == self.chi[-1]:
            self.M.append(m)
            self.chi.append(l2)


if __name__ == "__main__":
    s = MPS((1, 100, 1))
    s.M[1][0, :, :] = array([[0, 2 + 1j], [1, 0]])
    s.M[2][:, :, 0] = array([[5 - 0.3j, 2], [0.3, 4.3]])
    np.set_printoptions(precision=5)
    s.vidalize()
    print(s.M)
    print(s.lamb)
