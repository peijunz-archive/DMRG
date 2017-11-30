'''Matrix product state class

Infrastructure for DMRG and iTEBD

'''
import math
import numpy as np
import scipy.linalg as la
import copy

nx = np.newaxis


def cut(s, x, err):
    s /= s[0]
    ind = sum(np.abs(s) > err)
    ind = min(ind, x)
    s = s[:ind]
    s /= la.norm(s)
    return s, ind


def svd_cut(m, x=20, err=1e-10):
    u, s, v = la.svd(m, full_matrices=False)
    s, k = cut(s, x, err)
    return u[:, :k], s, v[:k]


class State:
    '''Length:
    + o --- L+1: 0~L
    + x --- L+1: 0~L
    + v --- L+2: (-1), 0, 1,..., L-1, (L)

    TODO Periodic boundary condition?
    '''

    def __init__(self, arg=(1, 1), dim=2, trun=20):
        '''
        arg may be x or M, only one of them is needed.
        x is an array of truncation length with list length n+1,
        so we have x[0]...x[n] and has shape (x[i], x[i+1])
        for M[i] where i is in range(0, n).

        Basically, if we are discussing MPS, x[0]=x[n]=1. But if we keep
        some boundaries open, for example x[n] > 1, then there is a series of MPS,
        which may be used to construct some other MPS by contraction.
        '''
        #self.dim = dim
        if isinstance(arg[0], np.ndarray):
            self.from_M(arg)
        else:
            self.from_x(arg, dim)
        self.M.append(np.eye(self.x[-1])[:, :, nx])
        self.M.append(np.eye(self.x[0])[nx])
        self.s = np.zeros_like(self.x, dtype='object')
        self.s[0] = np.ones(self.x[0])
        self.canonical = False
        self.trun = trun

    def from_x(self, x, dim):
        self.dim = dim
        self.x = np.array(x, dtype='i8')
        self.L = len(x) - 1
        self.M = [np.zeros(self.shape(i), 'complex') for i in range(self.L)]

    def from_M(self, M):
        L = [m.shape[0] for m in M]
        R = [m.shape[-1] for m in M]
        d = [m.shape[1] for m in M]
        assert L[1:] == R[:-1], "Incompatible matrices!"
        for i in d:
            assert i == d[0], "Incompatible shape!"
        self.dim = d[0]
        L.append(R[-1])
        self.x = np.array(L, dtype='i8')
        self.M = [np.array(i, 'complex') for i in M]
        self.L = len(M)

    @staticmethod
    def naive(*L):
        '''L is N*s matrix, x are ones'''
        if len(L) == 1:
            L = L[0]
        if isinstance(L[0], int):
            L = [(1, -1) if np.isinf(i) else (i, 1-i) for i in L]
        L = np.array(L)[:, nx, :, nx]
        return State(L)

    def __getitem__(self, ind):
        '''(1-i)/i=1/i-1=a*exp(it), i=1/(1+a*exp(it))'''
        return State.naive(ind).dot(self)

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
    def Sl(self):
        return self.s[:-1]

    @property
    def Sr(self):
        return self.s[1:]

    @property
    def xl(self):
        return self.x[:-1]

    @property
    def xr(self):
        return self.x[1:]

    def ortho_left_site(self, i):
        '''Left orthogonalization'''
        # Preparing work
        before = (-1, self.xr[i])
        after = (self.xl[i], self.dim, -1)
        self.M[i] *= self.Sl[i][:, nx, nx]
        # SVD
        u, s, v = svd_cut(self.M[i].reshape(before), self.trun)
        # Post job
        self.M[i] = u.reshape(after)
        self.xr[i] = len(s)
        self.Sr[i] = s
        self.M[i + 1] = np.einsum('al, lcr->acr', v, self.M[i + 1])

    def ortho_right_site(self, i):
        '''Right orthogonalization'''
        # Preparing work
        before = (self.xl[i], -1)
        after = (-1, self.dim, self.xr[i])
        self.M[i] *= self.Sr[i][nx, nx]
        # SVD
        u, s, v = svd_cut(self.M[i].reshape(before), self.trun)
        # Post job
        self.M[i] = v.reshape(after)
        self.xl[i] = len(s)
        self.Sl[i] = s
        self.M[i - 1] = np.einsum('lcr, ra->lca', self.M[i - 1], u)

    def unify_end(self):
        '''Remove possible exp(it) in auxillary M[-1] and M[L]'''
        if self.xl[0] == 1:
            self.M[0] /= self.M[-1][0, 0, 0]
            self.M[-1][0, 0, 0] = 1
        if self.xr[-1] == 1:
            self.M[self.L - 1] /= self.M[self.L][0, 0, 0]
            self.M[self.L][0, 0, 0] = 1

    def canon(self):
        '''Decompose MPS into ΓΛΓΛΓΛΓΛΓ'''
        for i in range(self.L):
            self.ortho_left_site(i)
        for i in reversed(range(self.L)):
            self.ortho_right_site(i)
        self.unify_end()
        self.canonical = True

    def B(self, i):
        '''Matrix after Orthonormalization'''
        return self.M[i]

    def block_single(self, i):
        '''Center Matrix with both circles'''
        return self.Sl[i][:, nx, nx] * self.M[i]

    def block(self, start, n):
        s = self.block_single(start)
        for i in range(start+1, start+n):
            s = np.einsum('abc, cde->abde', s, self.B(i))
            s = s.reshape(s.shape[0], -1, s.shape[-1])
        return s

    def verify_shape(self):
        '''Verify that x is compatible with M'''
        # print(self.M)
        for i in range(self.L):
            expt = self.shape(i)
            got = self.M[i].shape
            m = 'Shape error at {}, expected {}, got {}'.format(i, expt, got)
            assert expt == got, m

    def dot(self, rhs):
        '''Inner product between two wavefunctions'''
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, lcj->rj', self.block_single(0).conj(), rhs.block_single(0))
        for i in range(1, self.L):
            E = np.einsum('kl, kno, lnr->or', E, self.B(i).conj(), rhs.B(i))
        return np.trace(E)

    def corr(self, *ops):
        '''oplist should be a list of ordered operators applied sequentially like
        (4, sigma[3]), (6, sigma[2]), (7, sigma[1]), which means
        E[Z_4*Y_6*X_7]
        '''
        assert(self.canonical)
        D = dict(ops)
        start, end = min(D.keys()), max(D.keys())+1
        E = np.diag(self.Sl[start]**2)
        for i in range(start, end):
            b=self.B(i)
            if i in D:
                E = np.einsum('kl, kio, ij, ljr->or', E, b.conj(), D[i], b)
            else:
                E = np.einsum('kl, kio, lir->or', E, b.conj(), b)
        return np.trace(E)

    def measure(self, start, op, Hermitian=True):
        n = round(math.log(op.size, self.dim)/2)
        s = self.block(start, n)
        ret = np.einsum('abd, aed, be', s, s.conj(), op)
        return ret.real if Hermitian else ret

    def update_single(self, U, i, unitary=True):
        '''Apply U at single site i'''
        self.M[i] = np.einsum('lcr, dc->ldr', self.M[i], U)
        if not unitary:
            self.ortho_left_site(i)

    def update_double(self, U, i, unitary=True):
        '''Double site i, i+1 ?unitary update

        + For unitary update, there is no need to affect boundary.
        + For virtual time, exp(-tau*H) is no longer unitary. Update site i+1
            + For non unitary case, if we update from left to right, then we
              can easily make it left orthogonalized, but not right.
            + After one run from L to R, we should R-orthogonalize the chain.'''
        mb = np.einsum('lcr, rjk, abcj->labk', self.B(i), self.B(i + 1), U)
        m = self.Sl[i][:, nx, nx, nx] * mb
        sh = m.shape
        u, s, v = svd_cut(m.reshape(sh[0] * sh[1], -1), self.trun)
        self.xr[i] = len(s)
        u = u.reshape(*sh[:2], -1)
        v = v.reshape(-1, *sh[2:])
        self.Sr[i] = s
        self.M[i] = np.einsum('ijkl, mkl->ijm', mb, v.conj())
        self.M[i + 1] = v
        if not unitary:
            self.ortho_left_site(i + 1)

    def update_k(self, U, i, k, unitary=True):
        '''General multiple sites update'''
        pass

    def run_from(self, start, U, dt):
        for i in range(start, self.L - 1, 2):
            self.update_double(U, i)

    def iTEBD_double(self, H, t, n=100):
        '''Second Order Suzuki Trotter Expansion'''
        dt = t / n
        U = la.expm(-1j * H * dt).reshape([self.dim] * 4)
        self.run_from(0, U, dt / 2)
        for i in range(n - 1):
            self.run_from(1, U, dt)
            self.run_from(0, U, dt)
        self.run_from(1, U, dt)
        self.run_from(0, U, dt / 2)

    def copy(self):
        return copy.deepcopy(self)

    def testCanon(self, err=1e-13):
        for i in range(self.L):
            S = self.block_single(i)
            TSS = np.einsum("ijk, ijn->kn", S.conj(), S)
            SST = np.einsum("ijk, ljk->il", S, S.conj())
            assert la.norm(TSS - np.diag(self.Sr[i])**2) < err
            assert la.norm(SST - np.diag(self.Sl[i])**2) < err
