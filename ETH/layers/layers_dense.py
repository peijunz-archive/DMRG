from .layers import Layers
import numpy as np
from typing import Tuple, List


def transform(op, U, sh0):
    """
    Transform operator with single local :math:`U_{ij}` to slots determined
    by sh0

    Parameters
    ----------
        op: np.ndarray
            operator to be transformed
        U: np.ndarray
            Unitary transformation, should be a square matrix
        sh0: int
            zeroth elem of shape, determines starting slot

    Returns
    --------
        op: np.ndarray
            The operator after transformation
    """
    return np.einsum("ij, kjl->kil", U, op.reshape(sh0, U.shape[0], -1))


class LayersDense(Layers):
    """Contraction of layers to rho and H in full :math:`2^L \\times 2^L` form

    Important conventions or notes:
        + Unitaries are plugged into rho and H as :math:`\\mathrm{tr}[U \\rho U^+ H]`
        + U with smaller layer index is closer to rho and farther from H

    Attributes
    ----------
        rho : np.ndarray
            Initial density matrix
        H: np.ndarray
            Hamiltonian
        H2: np.ndarray
            H^2
        L: int
            Chain length

    """

    def __init__(self, rho, H=None, D=4, dim=2, H2=None):
        """
        Parameters
        -----------
            rho: np.ndarray
                Density matrix
            H: np.ndarray, optional
                Hamiltonian or any Hermitian operator
            D: int, optional
                Depth of circuit
            dim: int, optional
                Hilbert space dimension of single site
            H2: np.ndarray, optional
                H^2 matrix an be generated from H if it is None
        """
        self.rho = rho
        self.H = H
        if H2 is None:
            H2 = H@H
        self.H2 = H2
        self.L = np.int(np.log2(self.H2.size)/np.log2(dim**2))
        super().__init__(self.L-1, D)
        for i in self.traversal():
            self.U[i] = np.eye(4)

    def apply_single(self, ind, op, hc=False):
        """Apply single local unitary to operator op

        Parameters
        -----------
            inds: tuple
                Index of Unitary to apply
            op: np.ndarray
                operator to contract
            hc: bool
                Hermitian conjugate
        Returns
        --------
            op: ndarray
                The new operator after application of U
        """
        if hc:
            U = self[ind].T.conj()
        else:
            U = self[ind]
        op = transform(op, U, 2**ind[1])
        op = transform(op, U.conj(), 2**(ind[1]+self.L))
        return op

    def apply_list(self, inds, op, hc=False):
        """
        Apply multiple local unitaries to operator op

        Parameters
        -----------
            inds: list
                List of indices of unitaries
            op: np.ndarray
                operator to contract
            hc: bool
                Hermitian conjugate. Apply :math:`U_1^+U_2^+U_3^+` rather
                than :math:`U_3U_2U_1`

        Returns
        ----------
            op: np.ndarray
                Transformed operator
        """
        if hc:
            inds = inds[::-1]
        for ind in inds:
            op = self.apply_single(ind, op, hc)
        return op

    def contract_rho(self):
        """Contract all unitaries to rho

        Returns
        -------
            rho: np.ndarray
        """
        rho = self.apply_list(self.indices, self.rho)
        return rho.reshape((2**self.L,)*2)

    def contract_hole(self, op1, op2, ind, apply=True, hc=True):
        """In :math:`U\\rho U^+ H`, contract all irrelevant index between
        rho and H that is not connected to U[ind].

        Parameters
        -----
            op1: np.ndarray
                Operator O1, typically rho
            op2: np.ndarray
                Operator O2, typically H
            ind: Tuple[int, int]
                the hole to preserve
            apply: bool
                If True, apply U at ind with op2
            hc: bool
                Hermitian conjugate. Use U.T.conj() instead of U
        Returns
        --------
            hole: np.ndarray
                Dangling legs after partial contraction of O1 and O2
        Note
        ------
        we are optimizing rho side U, so U at ind is contracted with H
        """
        sh = (2**ind[1], 4, 2**(self.L - ind[1] - 2))*2
        if apply:
            op2 = self.apply_single(ind, op2, hc=hc)
        return np.einsum('lonipk, ijklmn->mopj', op1.reshape(sh), op2.reshape(sh))

    # @profile
    def contract_naive(self, H, i):
        """Contract lower part of U to rho, upper part to H. Keep the hole
        corresponding to i-th unitary.        Parameters
        ------
            H: np.ndarray
                Hermitian, typically H or H^2
            i: int
                The absolute index of targeting unitary
        Returns
        -----
            hole: np.ndarray
                Hole for i-th unitary

        Note
        -----
            This routine is not efficient at all. It is only used to
            test that more efficient but complicated routine is correct.

        """
        rho = self.apply_list(self.indices[:i], self.rho)
        H = self.apply_list(self.indices[i+1:], H, hc=True)
        return self.contract_hole(rho, H, self.indices[i])

    def contract_cycle_for(self, *ops):
        """Forward Hole contraction Cycle

        Contract all to Hermitian as starting point, optimize rho side

        Parameters
        -----
            ops: np.ndarray
                List of Hermitian that needs to contract with rho to generate holes
        Returns
        -----
            ind: Tuple[int, int]
                index of hole
            *holes: np.ndarray
                Holes corresponds to ops
        """
        rho = self.rho
        ops = [self.apply_list(self.indices, op, hc=True) for op in ops]

        for l, mid in zip([None]+self.indices[:-1], self.indices):
            # March to new U
            if l:
                rho = self.apply_single(l, rho)
            # Contract
            V = [self.contract_hole(rho, H, mid, apply=False) for H in ops]
            # Retreat
            for i in range(len(ops)):
                ops[i] = self.apply_single(mid, ops[i])
            yield (mid, *V)

    def contract_cycle_back(self, *ops):
        """Forward Hole contraction Cycle

        Contract all to rho as starting point, optimize Hermitian side

        Parameters
        -----
            ops: np.ndarray
                List of Hermitian that needs to contract with rho to generate holes
        Returns
        -----
            ind: Tuple[int, int]
                index of hole
            *holes: np.ndarray
                Holes corresponds to ops
        """
        ops = list(ops)
        rho = self.apply_list(self.indices, self.rho)

        for mid, r in zip(self.indices[::-1], [None]+self.indices[1:][::-1]):
            # March to new U
            if r:
                for i in range(len(ops)):
                    ops[i] = self.apply_single(r, ops[i], hc=True)
            # Contract
            V = [self.contract_hole(H, rho, mid, apply=False) for H in ops]
            # Retreat
            rho = self.apply_single(mid, rho, hc=True)
            yield (mid, *V)

    def contract_cycle(self, *ops, back=False):
        '''Wrapper for forward/backward contraction cycle

        Args
        ----
        back: bool
            Use backward cycle if True. Default is forward.

        Returns
        ----
            generator
        '''
        if back:
            return self.contract_cycle_back(*ops)
        else:
            return self.contract_cycle_for(*ops)
