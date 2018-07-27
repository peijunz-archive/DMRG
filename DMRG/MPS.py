"""Matrix product state class

Infrastructure for DMRG and iTEBD

"""
import math
import numpy as np
import scipy.linalg as la
import copy
from . import ST
from typing import List


def truncate(s, trun, err):
    """
    Truncate s based on trun or err, whichever results in smaller bond dimension.

    Args
    ----
        s: np.ndarray
            The eigenvalues of density matrix that need to be truncated
        trun: int
            The max bond dimension to preserve
        err: float
             If an eigenvalue is smaller than err * max_eigval, it would be truncated
    Returns
    ----
        np.ndarray:
            Truncated eigenvalues
        int:
            Number of surviving eigenvalues
    """
    s /= s[0]
    ind = sum(np.abs(s) > err)
    ind = min(ind, trun)
    s = s[:ind]
    s /= la.norm(s)
    return s, ind


def svd_truncate(m, trun=20, err=1e-10):
    """Do SVD and then truncate with parameters trun and err"""
    u, s, v = la.svd(m, full_matrices=False)
    s, k = truncate(s, trun, err)
    return u[:, :k], s, v[:k]


class MPS:
    """Matrix Product state

    Attributes
    ----------
    s: np.ndarray
        Eigenvalues in each bond, i.e. circles. Use MPS.Sl or MPS.Sr
        to get the circle in the left/right of a site
    M: np.ndarray
        list of local tensors, i.e. Triangles
    x: np.ndarray
        Bond dimensions. Use MPS.xl/MPS.xr to get the left/right of
        a site
    dim: int
        dimension for local site
    L: int
        Length of chain
    trun: int
        Refer to truncate
    err: float
        Refer to truncate


    Notes
    -----
    Example of MPS

    .. code::
        1 -- v -- 2 -- v -- 2 -- v -- 1
             |         |         |
             2         2         2

    + dim is 2 for all sites
    + Number of sites(represented by v) is 3
    + Bond dimension chi = (1, 2, 2, 1)
        + For a Closed MPS, x[0]=x[n]=1. But if we keep some boundaries open,
          for example x[n] > 1, then there is a series of MPS, which
          may be used to construct some other MPS by contraction.
    + Local tensors has shapes [(1, 2, 2), (2, 2, 2), (2, 2, 1)]

    Physical data structure
    ^^^^^^^^^^^^^^^^^^^^^^^

    Length of every element:
        + Circles o --- L+1: 0~L
        + Bonds x(chi) --- L+1: 0~L
        + Sites --- L+2: (-1), 0, 1,..., L-1, (L)
            + There are two dummy sites at -1 and L to ease coding

    Canonical form
    ^^^^^^^^^^^^^^^

    It is in B form, where M includes the right circle

    """

    def __init__(self, arg=(1, 1), dim=2, trun=20, err=1e-6):
        """
        Args
        ---
            arg:
                It may be x(chi) or M, only one of them is needed.
                + x (chi) is an array of bond dimension with list length n+1,
                      so we have x[0]...x[n] and has shape (x[i], x[i+1]) for M[i] where i is in range(0, n)
                + M is the actual MPS
            dim: int
                Hilbert space dimension for each site
            trun: int
                Refer to truncate
            err: float
                Refer to truncate
        """
        if isinstance(arg[0], np.ndarray):
            self.from_M(arg)
        else:
            self.from_chi(arg, dim)
        self.M.append(np.eye(self.x[-1])[:, :, np.newaxis])
        self.M.append(np.eye(self.x[0])[np.newaxis])
        self.s = np.zeros_like(self.x, dtype='object')
        self.s[0] = np.ones(self.x[0])
        self.canonical = False
        self.trun = trun
        self.err = err

    def from_chi(self, x, dim):
        """Construct MPS from chi

        Args
        ---
            x: np.ndarray
                1 dimensional array specifying bond dimensions
            dim: int
        """
        self.dim = dim
        self.x = np.array(x, dtype='i8')
        self.L = len(x) - 1
        self.M = [np.zeros(self.shape(i), 'complex') for i in range(self.L)]

    def from_M(self, M):
        """Construct MPS from M

        Notes
        -----
        Every element of M is a local tensors T_{ijk}:
        .. code::
            i ---- T ---- k
                   |
                   j

        Args
        ---
            M: List[np.ndarray]
        """
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
    def naive(*local):
        """
        Generate MPS (without entanglement entropy) from local states

        Args
        ---
            local:
                list of local states
        """
        if len(local) == 1:
            local = local[0]
        if isinstance(local[0], int):
            local = [(i, 1-i) for i in local]
        local = np.array(local)[:, np.newaxis, :, np.newaxis]
        return MPS(local)

    def __getitem__(self, ind)->float:
        """Calculate product to local MPS generated by ind
        Args:
            ind: list of density matrix at each site
        """
        return self.naive(ind).dot(self)

    def shape(self, i):
        """Shape of the i-th MPS matrix

        Args
        ---
            i: int
                Index of site
        """
        return self.xl[i], self.dim, self.xr[i]

    def graph(self):
        """
        Generate graph of MPS like:
        .. code::
            1 -- v -- 2 -- v -- 1
                 |         |
                 2         2
        """
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
        """Left circles"""
        return self.s[:-1]

    @property
    def Sr(self):
        """Right Circles"""
        return self.s[1:]

    @property
    def xl(self):
        """Left bond dimensions"""
        return self.x[:-1]

    @property
    def xr(self):
        """Right bond dimensions"""
        return self.x[1:]

    def ortho_left_site(self, i):
        """Left orthogonalization

        Args
        ---
            i: int
                Index of site
        """
        # Preparing
        before = (-1, self.xr[i])
        after = (self.xl[i], self.dim, -1)
        self.M[i] *= self.Sl[i][:, np.newaxis, np.newaxis]
        # SVD
        u, s, v = svd_truncate(self.M[i].reshape(before), self.trun, self.err)
        # Ending
        self.M[i] = u.reshape(after)
        self.xr[i] = len(s)
        self.Sr[i] = s
        self.M[i + 1] = np.einsum('al, lcr->acr', v, self.M[i + 1])

    def ortho_right_site(self, i):
        """Right orthogonalization

        Args
        ---
            i: int
                Index of site
        """
        # Preparing
        before = (self.xl[i], -1)
        after = (-1, self.dim, self.xr[i])
        self.M[i] *= self.Sr[i][np.newaxis, np.newaxis]
        # SVD
        u, s, v = svd_truncate(self.M[i].reshape(before), self.trun, self.err)
        # Ending
        self.M[i] = v.reshape(after)
        self.xl[i] = len(s)
        self.Sl[i] = s
        self.M[i - 1] = np.einsum('lcr, ra->lca', self.M[i - 1], u)

    def unify_end(self):
        """Remove possible exp(it) in auxiliary M[-1] and M[L]"""
        if self.xl[0] == 1:
            self.M[0] /= self.M[-1][0, 0, 0]
            self.M[-1][0, 0, 0] = 1
        if self.xr[-1] == 1:
            self.M[self.L - 1] /= self.M[self.L][0, 0, 0]
            self.M[self.L][0, 0, 0] = 1

    def canon(self):
        """Decompose MPS into ΓΛΓΛΓΛΓΛΓ"""
        for i in range(self.L):
            self.ortho_left_site(i)
        for i in reversed(range(self.L)):
            self.ortho_right_site(i)
        self.unify_end()
        self.canonical = True

    def B(self, i):
        """B Matrix after Orthonormalization"""
        return self.M[i]

    def block_single(self, i):
        """Matrix at a site with both circles, i.e. Combine Left circle and B matrix"""
        return self.Sl[i][:, np.newaxis, np.newaxis] * self.M[i]

    def block(self, start, n):
        """MPS state for sites in the middle

        Args
        ---
            start: int
                Starting site
            n: int
                Number of sites
        Returns
        ---
            MPS state for sites [start, start+n). It has included circle in the edge
        """
        s = self.block_single(start)
        for i in range(start+1, start+n):
            s = np.einsum('abc, cde->abde', s, self.B(i))
            s = s.reshape(s.shape[0], -1, s.shape[-1])
        return s

    def verify_shape(self):
        """Verify that x is compatible with M"""
        for i in range(self.L):
            expt = self.shape(i)
            got = self.M[i].shape
            m = 'Shape error at {}, expected {}, got {}'.format(i, expt, got)
            assert expt == got, m

    def dot(self, rhs):
        """Inner product between two wavefunctions

        Args
        ---
            rhs: MPS
        Returns
        ---
            float:
                inner product
        """
        assert self.dim == rhs.dim, "Shape conflict between states"
        E = np.einsum('lcr, lcj->rj', self.block_single(0).conj(),
                      rhs.block_single(0))
        for i in range(1, self.L):
            E = np.einsum('kl, kno, lnr->or', E, self.B(i).conj(), rhs.B(i))
        return np.trace(E)

    # @profile
    def corr(self, *ops):
        """Correlation function of operators

        Args
        ----
            *ops: Tuple[int, np.ndarray]
                Every element is a tuple of site index and operators applied in it.
                They should passed into corr in index sorted order.
        Returns
        ---
            float:
                Correlation function

        Examples
        ---
        ops = [(4, sigma[3]), (6, sigma[2]), (7, sigma[1])]
        """
        assert(self.canonical)
        D = dict(ops)
        if len(D.keys()) == 0:
            return 1
        start, end = min(D.keys()), max(D.keys())+1
        E = np.diag(self.Sl[start]**2)
        for i in range(start, end):
            b = self.B(i)
            if i in D:
                E = np.einsum('kl, kio, ij, ljr->or', E,
                              b.conj(), D[i], b, optimize=False)
            else:
                E = np.einsum('kl, kio, lir->or', E,
                              b.conj(), b, optimize=False)
        return np.trace(E).real

    def measure(self, start, op, Hermitian=True):
        """Measure a operator over several sites

        Args
        ----
            start: int
                starting site
            op: np.ndarray
                The multiple sites operater
            Hermitian: bool
                Pass True if the operator is Hermitian
        Returns
        ----
            float:
                Expectation value of operator
        """
        n = round(math.log(op.size, self.dim)/2)
        s = self.block(start, n)
        ret = np.einsum('abd, aed, be', s, s.conj(), op, optimize=False)
        return ret.real if Hermitian else ret

    # @profile
    def update_single(self, U, i, unitary=True):
        """Single site update
        Args:
            U: np.ndarray
                Single site transformation operator
            i: int
                site index
            unitary:
                Unitarity of U. For virtual time iTEBD, it is not unitary.
        """
        self.M[i] = np.einsum('lcr, dc->ldr', self.M[i], U)
        if not unitary:
            self.ortho_left_site(i)

    # @profile
    def update_double(self, U, i, unitary=True):
        """Double site update
        Args:
            U: np.ndarray
                Single site transformation operator
            i: int
                index of left site
            unitary:
                Unitarity of U. For virtual time iTEBD, it is not unitary.

        Notes
        -----
        + For unitary update, there is no need to affect boundary.
        + For virtual time, exp(-tau*H) is no longer unitary. Update site i+1
            + For non unitary case, if we update from left to right, then we
              can easily make it left orthogonalized, but not right.
            + After one run from L to R, we should R-orthogonalize the chain."""
        mb = np.einsum('lcr, rjk, abcj->labk', self.B(i), self.B(i + 1), U)
        m = self.Sl[i][:, np.newaxis, np.newaxis, np.newaxis] * mb
        sh = m.shape
        u, s, v = svd_truncate(
            m.reshape(sh[0] * sh[1], -1), self.trun, self.err)
        self.xr[i] = len(s)
        u = u.reshape(*sh[:2], -1)
        v = v.reshape(-1, *sh[2:])
        self.Sr[i] = s
        self.M[i] = np.einsum('ijkl, mkl->ijm', mb, v.conj())
        self.M[i + 1] = v
        if not unitary:
            self.ortho_left_site(i + 1)

    def update_k(self, U, i, k, unitary=True):
        """General multiple sites update"""
        raise NotImplementedError

    def iTEBD_double(self, H, t, n):
        """iTEBD algorithm for nearest interaction

        Args
        ---
            H: np.ndarray
                Two site interaction Hamiltonian
            t: float
                Time length for evolution
            n: int
                Number of time slices
        """
        def even_update(k):
            U = la.expm(-1j * H * k * t).reshape([self.dim] * 4)

            def _even_update():
                for i in range(0, self.L - 1, 2):
                    self.update_double(U, i)
            return _even_update

        def odd_update(k):
            U = la.expm(-1j * H * k * t).reshape([self.dim] * 4)

            def _odd_update():
                for i in range(1, self.L - 1, 2):
                    self.update_double(U, i)
            return _odd_update
        ST.ST2((even_update, odd_update), n)

    def copy(self):
        return copy.deepcopy(self)

    def testCanon(self, err=1e-13):
        """Test that the MPS is really in canonical form

        Args
        ----
        err: float
            Error tolerance
        """
        for i in range(self.L):
            S = self.block_single(i)
            TSS = np.einsum("ijk, ijn->kn", S.conj(), S)
            SST = np.einsum("ijk, ljk->il", S, S.conj())
            assert la.norm(TSS - np.diag(self.Sr[i])**2) < err
            assert la.norm(SST - np.diag(self.Sl[i])**2) < err
