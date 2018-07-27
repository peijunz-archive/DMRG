from .layers import Layers
import numpy as np


def transform(op, U, sh0):
    '''Transform operator op with single U_{ij} to slots determined
    by sh0
    Args:
        + op    operator to be transformed
        + U     Unitary transformation, should be a square matrix
        + sh0   zeroth elem of shape, determines starting slot
    Return:
        The operator after transformation
    '''
    return np.einsum("ij, kjl->kil", U, op.reshape(sh0, U.shape[0], -1))


class LayersDense(Layers):
    '''Density matrix and observables are represented in full form (having shape 2^L by 2^L)
    Important conventions or notes:
    + Unitaries are plugged into rho and H as tr[U rho U^+ H]
    + 
    '''

    def __init__(self, rho, H=None, D=4, dim=2, H2=None):
        '''
        Args:
            rho     Density matrix
        '''
        self.rho = rho
        self.H = H
        if H2 is None:
            H2 = H@H
        self.H2 = H2
        self.L = np.int(np.log2(self.H2.size)/np.log2(dim**2))
        super().__init__(self.L-1, D)
        for i in self.visit_all():
            self.U[i] = np.eye(4)

    def apply_single(self, ind, op, hc=False):
        '''
        Apply single local unitary to operator op
        Args:
            + inds  Index of Unitary
            + op    operator to contract
            + hc    Hermitian conjugate of U3U2U1, which gives U1+U2+U3+'''
        if hc:
            U = self[ind].T.conj()
        else:
            U = self[ind]
        op = transform(op, U, 2**ind[1])
        op = transform(op, U.conj(), 2**(ind[1]+self.L))
        return op

    def apply_list(self, inds, op, hc=False):
        '''
        Apply multiple local unitaries to operator op
        Args:
            + inds  List of unitaries
            + op    operator to contract
            + hc    Hermitian conjugate of U3U2U1, which gives U1+U2+U3+'''
        if hc:
            inds = inds[::-1]
        for ind in inds:
            op = self.apply_single(ind, op, hc)
        return op

    def contract_rho(self):
        '''Contract all unitaries to rho'''
        rho = self.apply_list(self.indices, self.rho)
        return rho.reshape((2**self.L,)*2)

    def contract_hole(self, op1, op2, ind, middle=True, hc=True):
        '''
        In anology of U_{ij} O1_{jk} U+_{kl} O2_{li}. But we have extra legs
        Args:
            + op1   Operator in analogy of O1
            + op2   Operator in analogy of O2
            + ind   the hole to preserve
            + middle    if True, contract U at ind with op2
            + hc    Use U.T.conj() instead of U
        '''
        sh = (2**ind[1], 4, 2**(self.L - ind[1] - 2))*2
        '''Convention: we are optimizing rho side U, so U at ind is contracted with H'''
        if middle:
            op2 = self.apply_single(ind, op2, hc=hc)
        return np.einsum('lonipk, ijklmn->mopj', op1.reshape(sh), op2.reshape(sh))

    # @profile
    def contract_naive(self, H, i):
        '''Contract lower part of U to rho, upper part to H
        Only for testing purpose'''
        rho = self.apply_list(self.indices[:i], self.rho)
        H = self.apply_list(self.indices[i+1:], H, hc=True)
        return self.contract_hole(rho, H, self.indices[i])

    def contract_cycle_for(self, *ops):
        '''Forward: Contract all to operators as initial, optimize rho side'''
        rho = self.rho
        ops = [self.apply_list(self.indices, op, hc=True) for op in ops]

        for l, mid in zip([None]+self.indices[:-1], self.indices):
            # March to new U
            if l:
                rho = self.apply_single(l, rho)
            # Contract
            V = [self.contract_hole(rho, H, mid, middle=False) for H in ops]
            # Retreat
            for i in range(len(ops)):
                ops[i] = self.apply_single(mid, ops[i])
            yield (mid, *V)

    def contract_cycle_back(self, *ops):
        '''Backward: Contract all to rho as initial, optimize H side'''
        ops = list(ops)
        rho = self.apply_list(self.indices, self.rho)

        for mid, r in zip(self.indices[::-1], [None]+self.indices[1:][::-1]):
            # March to new U
            if r:
                for i in range(len(ops)):
                    ops[i] = self.apply_single(r, ops[i], hc=True)
            # Contract
            V = [self.contract_hole(H, rho, mid, middle=False) for H in ops]
            # Retreat
            rho = self.apply_single(mid, rho, hc=True)
            yield (mid, *V)

    def contract_cycle(self, *ops, back=False):
        if back:
            return self.contract_cycle_back(*ops)
        else:
            return self.contract_cycle_for(*ops)
