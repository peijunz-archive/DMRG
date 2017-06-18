'''Matrix product state class

Infrastructure for DMRG and iTEBD

'''
import numpy as np
import scipy.linalg as la
from functools import reduce

def svd_cut(m, err=1e-16, chi=20):
    u, s, v = la.svd(m, full_matrices=False)
    ind = sum(np.abs(s) > err)
    ind = min(ind, chi)
    u = u[:, :ind]
    v = v[:ind]
    s = s[:ind]
    return u, s, v

class State:
    '''Length:
    + o --- L+1: 0~L
    + chi --- L+1: 0~L
    + v --- L+2: (-1), 0, 1,..., L-1, (L)
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
        self.M.append(np.eye(self.chi[-1], dtype='complex').reshape(self.chi[-1], -1, 1))
        self.M.append(np.eye(self.chi[0], dtype='complex').reshape(1, self.chi[0], -1))
        self.rho = np.zeros_like(self.chi, dtype='object')
        self.canonical = False

    def from_chi(self, chi):
        self.chi = np.array(chi, dtype='i8')
        self.L = len(chi) - 1
        self.M = [np.zeros(self.shape(i), 'complex') for i in range(self.L)]

    def from_M(self, M):
        L = [m.shape[0] for m in M]
        R = [m.shape[-1] for m in M]
        assert L[1:] == R[:-1], "Incompatible matrices!"
        L.append(R[-1])
        self.chi = np.array(L, dtype='i8')
        self.M = [np.array(i, 'complex') for i in M]
        self.L = len(M)

    @staticmethod
    def naive(*L):
        '''L is N*s matrix, chi are ones'''
        L = np.array(L).reshape([-1, 1, 2, 1])
        return State(L)

    def shape(self, i):
        return self.xl[i], self.dim, self.xr[i]

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

    @property
    def rhol(self):
        return self.rho[:-1]

    @property
    def rhor(self):
        return self.rho[1:]

    @property
    def xl(self):
        return self.chi[:-1]

    @property
    def xr(self):
        return self.chi[1:]

    def ortho_left(self, cen=None):
        '''Go from the left end to left orthogonalize till center'''
        if cen is None:
            cen = self.L
        for i in range(cen):
            before = (-1, self.xr[i])
            after = (self.xl[i], self.dim, -1)
            u, s, v = svd_cut(self.M[i].reshape(before))
            self.xr[i] = len(s)
            self.M[i] = u.reshape(after)
            s /= la.norm(s)
            self.rhor[i] = s
            v *= s[:, np.newaxis]
            self.M[i + 1] = np.einsum('al, lcr->acr', v, self.M[i + 1])
        self.verify_shape()

    def ortho_right(self, cen=0):
        '''Go from the right end to right orthogonalize till center'''
        for i in reversed(range(cen, self.L)):
            before = (self.xl[i], -1)
            after = (-1, self.dim, self.xr[i])
            u, s, v = svd_cut(self.M[i].reshape(before))
            self.xl[i] = len(s)
            self.M[i] = v.reshape(after)
            s /= la.norm(s)
            self.rhol[i] = s
            u *= s[np.newaxis, :]
            self.M[i - 1] = np.einsum('lcr, ra->lca', self.M[i - 1], u)  # TODO
        self.verify_shape()
    def unify_end(self):
        if self.xl[0] == 1:
            self.M[0]/=self.M[-1][0,0,0]
            self.M[-1][0,0,0]= 1 
        if self.xr[-1] == 1:
            self.M[self.L-1]/=self.M[self.L][0,0,0]
            self.M[self.L][0,0,0]= 1
    def canon(self):
        '''Decompose MPS into ΓΛΓΛΓΛΓΛΓ'''
        self.ortho_left()
        self.ortho_right()
        for i in range(0, self.L - 1):
            self.M[i] /= self.rhor[i][np.newaxis, np.newaxis, :]
        self.unify_end()
        self.canonical = True

    def A(self, i):
        '''Left Matrix after Orthonormalization'''
        if i == 0:
            return self.M[0]
        else:
            return self.M[i] * self.rhol[i][:, np.newaxis, np.newaxis]

    def B(self, i):
        '''Left Matrix after Orthonormalization'''
        if i == self.L - 1:
            return self.M[-1]
        else:
            return self.M[i] * self.rhor[i][np.newaxis, np.newaxis, :]

    def verify_shape(self):
        '''Verify that chi is compatible with M'''
        print(self.M)
        for i in range(self.L):
            expect = self.shape(i)
            got = self.M[i].shape
            assert expect == got, 'Shape error at {}, expected {}, got {}'.format(
                i, expect, got)

    def dot(self, rhs):
        '''Inner product between two wavefunctions'''
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, lcj->rj', self.A(0).conj(), rhs.A(0))
        for i in range(1, self.L):
            E = np.einsum('kl, kno, lnr->or', E, self.A(i).conj(), rhs.A(i))
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
        E = np.eye(self.xl[x], dtype='complex')
        for i, op in ops:
            if x < i:
                for j in range(x, j):
                    E = np.einsum('kl, kio, lir->or',
                                  E, self.A(i).conj(), self.A(i))
            x = i + 1
            E = np.einsum('kl, kio, ij, ljr->or',
                          E, self.A(i).conj(), op, self.A(i))
        return np.einsum('ii, i', E, self.rhor[i]**2).real

    def update_su(self, U, site, unitary=True):
        '''Apply U at sites, there are 2*len(sites) axes in U

        + For unitary update, no need to affect boundary.
        + For virtual time, exp(-tau*H) is no longer unitary
        '''
        # unitary:
        self.M[site] = np.einsum('lcr, dc->ldr', self.M[site], U)
        return self

    def update_du(self, U, site, unitary=True, chi=20):
        '''Have not been tested'''
        m=np.einsum('lcr, r, rjk, abcj->labk',
                    self.A(site), self.rhor[site], self.B(site+1), U)
        sh=m.shape
        u, s, v=la.svd(m.reshape((sh[0]*sh[1], -1)))
        if len(s) > chi:
            u = u[:, :chi]
            v = v[:chi]
            s = s[:chi]
            self.xr[i] = ind
        u=u.reshape((*sh[:2], -1))
        v=v.reshape((-1, *sh[2:]))
        self.M[site] = u/self.rhol[site][:, np.newaxis, np.newaxis]
        self.M[site+1] = v/self.rhor[site+1][np.newaxis, np.newaxis, :]
        self.rhor[site]=s/la.norm(s)
        return self

def test1():
    s = State((1, 2, 2))
    s.M[0][0, :, :] = np.array([[5, 2 + 1j], [9j, 6j]])
    s.M[1][:, :, 0] = np.array([[5 - 1j, 2 - 3j], [0.3, 4]])
    s.M[1][:, :, 1] = np.array([[6, 3j], [5, 0]])
    np.set_printoptions(precision=5)
    print(s)
    s.canon()
    print(s.M, sep='\n')
    print(s.rho, sep='\n')
    print(s.dot(s))
    
if __name__ == "__main__":
    #from pauli import sigma
    #s = State.naive([0, 1], [1, 0], [2, 3])
    #s.canon()
    #print(s.dot(s))
    #t=np.pi/2
    #n=17
    #o=la.expm(t/n*1j*np.kron(sigma[3], sigma[3])).reshape([2,2,2,2])
    #print(o)
    #for i in range(n):
        #s.update_du(o, 0)
    #print(s.M)
    #print(s.rho)
    #print(s.dot(s))
    test1()
