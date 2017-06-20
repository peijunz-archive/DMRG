'''Matrix product state class

Infrastructure for DMRG and iTEBD

'''
import numpy as np
import scipy.linalg as la


def svd_cut(m, err=1e-16, x=20):
    u, s, v = la.svd(m, full_matrices=False)
    s /= la.norm(s)
    ind = sum(np.abs(s) > err)
    ind = min(ind, x)
    return u[:, :ind], s[:ind], v[:ind]


class State:
    '''Length:
    + o --- L+1: 0~L
    + x --- L+1: 0~L
    + v --- L+2: (-1), 0, 1,..., L-1, (L)'''

    def __init__(self, arg=(1, 1), dim=2):
        '''
        arg may be x or M, only one of them is needed.
        x is an array of truncation length with list length n+1,
        so we have x[0]...x[n] and has shape (x[i], x[i+1])
        for M[i] where i is in range(0, n).

        Basically, if we are discussing MPS, x[0]=x[n]=1. But if we keep
        some boundaries open, for example x[n] > 1, then there is a series of MPS,
        which may be used to construct some other MPS by contraction.
        '''
        self.dim = dim
        if isinstance(arg[0], np.ndarray):
            self.from_M(arg)
        else:
            self.from_x(arg)
        self.M.append(
            np.eye(self.x[-1], dtype='complex').reshape(self.x[-1], -1, 1))
        self.M.append(
            np.eye(self.x[0], dtype='complex').reshape(1, self.x[0], -1))
        self.s = np.zeros_like(self.x, dtype='object')
        for i in range(self.L+1):
            self.s[i]=np.ones(self.x[i], dtype='double')
        self.canonical = False

    def from_x(self, x):
        self.x = np.array(x, dtype='i8')
        self.L = len(x) - 1
        self.M = [np.zeros(self.shape(i), 'complex') for i in range(self.L)]

    def from_M(self, M):
        L = [m.shape[0] for m in M]
        R = [m.shape[-1] for m in M]
        assert L[1:] == R[:-1], "Incompatible matrices!"
        L.append(R[-1])
        self.x = np.array(L, dtype='i8')
        self.M = [np.array(i, 'complex') for i in M]
        self.L = len(M)

    @staticmethod
    def naive(*L):
        '''L is N*s matrix, x are ones'''
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
        s = '-- v --'.join('{:^3}'.format(i) for i in self.x)
        w = ''.join(['|' if i == 'v' else ' ' for i in s])
        n = w.replace('  |  ', '{:^5}'.format(self.dim))
        return '\n'.join((s, w, n))

    def __repr__(self):
        return self.graph()

    def __str__(self):
        return self.graph()

    @property
    def sl(self):
        return self.s[:-1]

    @property
    def sr(self):
        return self.s[1:]

    @property
    def xl(self):
        return self.x[:-1]

    @property
    def xr(self):
        return self.x[1:]

    def ortho_left(self, left=0, right=None):
        '''Left orthogonalization from the left end untill right end [)'''
        if right is None:
            right = self.L
        for i in range(left, right):
            before = (-1, self.xr[i])
            after = (self.xl[i], self.dim, -1)
            u, s, v = svd_cut(self.S(i).reshape(before))
            self.xr[i] = len(s)
            self.sr[i] = s
            self.M[i] = u.reshape(after) / self.sl[i][:, np.newaxis, np.newaxis]
            self.M[i + 1] = np.einsum('al, lcr->acr', v, self.M[i + 1])
        self.verify_shape()

    def ortho_right(self, right=None, left=-1):
        '''Right orthogonalization from the left end untill right end (]'''
        if right is None:
            right = self.L - 1
        for i in range(right, left, -1):
            before = (self.xl[i], -1)
            after = (-1, self.dim, self.xr[i])
            u, s, v = svd_cut(self.S(i).reshape(before))
            self.xl[i] = len(s)
            self.sl[i] = s
            self.M[i] = v.reshape(after) / self.sr[i][np.newaxis, np.newaxis, :]
            self.M[i - 1] = np.einsum('lcr, ra->lca', self.M[i - 1], u)
        self.verify_shape()

    def unify_end(self):
        if self.xl[0] == 1:
            self.M[0] /= self.M[-1][0, 0, 0]
            self.M[-1][0, 0, 0] = 1
        if self.xr[-1] == 1:
            self.M[self.L - 1] /= self.M[self.L][0, 0, 0]
            self.M[self.L][0, 0, 0] = 1

    def canon(self):
        '''Decompose MPS into ΓΛΓΛΓΛΓΛΓ'''
        self.ortho_left()
        self.ortho_right()
        #print(self.M)
        #print(self.s)
        #for i in range(0, self.L - 1):
            #self.M[i] /= self.sr[i][np.newaxis, np.newaxis, :]
        self.unify_end()
        self.canonical = True

    def Mat(self, i):
        if self.canonical:
            return self.A(i)
        else:
            return self.M[i]

    def A(self, i):
        '''Left Matrix after Orthonormalization'''
        #print(self.sl[i])
        return self.M[i] * self.sl[i][:, np.newaxis, np.newaxis]

    def B(self, i):
        '''Left Matrix after Orthonormalization'''
        return self.M[i] * self.sr[i][np.newaxis, np.newaxis, :]

    def S(self, i):
        '''Center Matrix with both circles'''
        return self.sl[i][:, np.newaxis, np.newaxis] * self.M[i] * self.sr[i][np.newaxis, np.newaxis, :]

    def verify_shape(self):
        '''Verify that x is compatible with M'''
        # print(self.M)
        for i in range(self.L):
            expect = self.shape(i)
            got = self.M[i].shape
            assert expect == got, 'Shape error at {}, expected {}, got {}'.format(
                i, expect, got)

    def dot(self, rhs):
        '''Inner product between two wavefunctions'''
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, lcj->rj', self.Mat(0).conj(), rhs.Mat(0))
        for i in range(1, self.L):
            E = np.einsum('kl, kno, lnr->or', E,
                          self.Mat(i).conj(), rhs.Mat(i))
        return np.trace(E)

    def __getitem__(self, ind):
        '''Bloch Wave function for each site'''
        return State.naive(ind).dot(self)

    def __add__(self, m):
        '''DEPRECATED'''
        l, d, r = m.shape
        if d == self.dim and l == self.x[-1]:
            self.M.append(m)
            self.x.append(l2)

    def measure(self, *ops):
        '''oplist should be a list of ordered operators applied sequentially like
        (4, sigma[3]), (6, sigma[2])

        TODO More complicated operators?
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
        return np.einsum('ii, i', E, self.sr[i]**2).real

    def update_single(self, U, site, unitary=True):
        '''Apply U at single site'''
        self.M[site] = np.einsum('lcr, dc->ldr', self.M[site], U)
        if not unitary:
            self.ortho_right(site, site-1)
            self.ortho_left(site, site+1)
        return self

    def update_double(self, U, site, unitary=True, x=20):
        '''Double site ?unitary update

        + For unitary update, no need to affect boundary.
        + For virtual time, exp(-tau*H) is no longer unitary'''
        m = np.einsum('lcr, r, rjk, abcj->labk',
                      self.A(site), self.sr[site], self.B(site + 1), U)
        sh = m.shape
        u, s, v = svd_cut(m.reshape((sh[0] * sh[1], -1)))
        self.xr[site] = len(s)
        u = u.reshape((*sh[:2], -1))
        v = v.reshape((-1, *sh[2:]))
        self.sr[site] = s
        if unitary:
            self.M[site] = u / self.sl[site][:, np.newaxis, np.newaxis]
            self.M[site + 1] = v / self.sr[site + 1][np.newaxis, np.newaxis, :]
        else:
            self.ortho_right(site, site-1)
            self.ortho_left(site+1, site+2)
            pass
        return self

    def slice():
        # TODO
        pass

    def copy():
        # TODO
        pass


def test_canon():
    s = State((1, 2, 1))
    s.M[0][0, :, :] = np.array([[1, 2 + 1j], [9j, 6j]])
    s.M[1][:, :, 0] = np.array([[- 1j, 2 - 3j], [0.3, 4]])
    #s.M[1][:, :, 1] = np.array([[5 - 1j, 2 - 3j], [0.3, 4]])
    np.set_printoptions(precision=5)
    print('Test Canon')
    print(s)
    s.canon()
    print(s.M, sep='\n')
    print(s.s, sep='\n')
    print(s.dot(s))


def two_body():
    '''For Hamiltonian, the eigenvec is ++, +-, -+, --'''
    from pauli import sigma
    print('Test two body')
    s = State.naive([1, 1], [1, 1])
    eig_states = [[[0, 1], [0, 1]],
                  [[0, 1], [1, 0]],
                  [[1, 0], [0, 1]],
                  [[1, 0], [1, 0]]]
    s.canon()
    t = np.pi
    n = 20
    o = la.expm(t / n * 1j * np.kron(sigma[3], sigma[3])).reshape([2, 2, 2, 2])
    for i in range(n + 1):
        L = np.array([s[eig_states[i]] for i in range(4)])
        print('i =', i)
        print(np.abs(L))
        print(np.angle(L))
        print('Measurement of Hamiltonian:', s.measure((0, sigma[3]), (1, sigma[3])))
        s.update_double(o, 0)


if __name__ == "__main__":
    test_canon()
    two_body()
