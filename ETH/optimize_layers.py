import numpy as np
import scipy.linalg as la
import ETH.optimization as opt


class Layers:
    def __init__(self, rho, H, D=4, L=None, dim=2):
        '''
        Data structure:
            Internally, local unitaries are stored in merged layers. Every layer contains
            multiple unitaries. Mergerd layer is a collection of even/odd paired layers.
            Its even/odd element is for even/odd layers, respectively.

            When indexing U, conceputual (layer height, x position) is used. The __getitem__
            will automatically convert it into self.U[layer_height//2]. The merged layer is
            organized in even layer first order.

        Args:
            + rho   density matrix
            + H     Hamiltonian
            + D     depth of layers
            + L     Length of the chain, if None, calculated from shape of rho and dim
            + dim   dim of each site Hilbert space

        Other Properties:
            + d     depth of merged layers.
            + l     number of elements for each merged layer
            + H2    Matrix Square of H
            + U     Tensor data structure storing merged layers and corresponding local Unitary
                    transformations
            + indexes   enumeration of all local unitaries
        '''
        self.D = D
        if L is None:
            self.L = np.int(np.log2(H.size)/np.log2(dim**2))
        else:
            self.L = L
        self.d = (self.D + 1) // 2
        self.l = self.L - 1
        self.rho = rho
        self.H = H
        self.H2 = H@H
        self.U = np.empty([self.d, self.l, 4, 4], dtype='complex')
        self.U[:] = np.eye(4)[np.newaxis, np.newaxis]
        self.indexes = list(self._visit_all())

    def __getitem__(self, ind):
        '''Get item by layer depth and starting position'''
        layer, pos = ind
        return self.U[layer // 2, pos]

    def __setitem__(self, ind, val):
        '''Get item by layer depth and starting position'''
        layer, pos = ind
        self.U[layer // 2, pos] = val

    def _visit_all(self):
        '''Enumerate all unitaries in order of dependency'''
        for d in range(self.d):
            # Even layers
            for i in np.arange(0, self.L - 1, 2):
                yield (2 * d, i)
            if 2 * d + 1 >= self.D:
                break
            # Odd layers
            for i in np.arange(1, self.L - 1, 2):
                yield (2 * d + 1, i)

    @staticmethod
    def transform_pair(op, U, i):
        '''Apply U to slot pair (i, i+1) of operator op
        Args:
            + op    operator to be transformed
            + U     Unitary transformation
            + i     index of slot i
        '''
        return np.einsum("ij, kjl->kil", U, op.reshape(2**i, 4, -1))

    def contract_op(self, inds, op, hc=False):
        '''
        Contract operator with given local unitaries
        Args:
            + inds  List of unitaries
            + op    operator to contract
            + hc    Hermitian conjugate of U3U2U1, which gives U1+U2+U3+'''
        if not hc:
            for ind in inds:
                op = self.transform_pair(op, self[ind], ind[1])
                op = self.transform_pair(op, self[ind].conj(), ind[1]+self.L)
        else:
            for ind in inds[::-1]:
                op = self.transform_pair(op, self[ind].T.conj(), ind[1])
                op = self.transform_pair(op, self[ind].T, ind[1]+self.L)
        return op

    def contract_rho(self):
        '''Contract all unitaries to rho'''
        rho = self.contract_op(self.indexes, self.rho)
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
            op2 = self.contract_op((ind,), op2, hc=hc)
        return np.einsum('lonipk, ijklmn->mopj', op1.reshape(sh), op2.reshape(sh))

    #@profile
    def contract_naive(self, H, i):
        '''Contract lower part of U to rho, upper part to H
        Only for testing purpose'''
        rho = self.contract_op(self.indexes[:i], self.rho)
        H = self.contract_op(self.indexes[i+1:], H, hc=True)
        return self.contract_hole(rho, H, self.indexes[i])

    def contract_cycle_for(self, *ops):
        '''Contract all to operators as initial, optimize rho side'''
        rho = self.rho
        ops = [self.contract_op(self.indexes, op, hc=True) for op in ops]

        for l, mid in zip([None]+self.indexes[:-1], self.indexes):
            # March to new U
            if l:
                rho = self.contract_op((l,), rho)
            # Contract
            V = [self.contract_hole(rho, H, mid, middle=False) for H in ops]
            # Retreat
            for i in range(len(ops)):
                ops[i] = self.contract_op((mid,), ops[i])
            yield (mid, *V)

    def contract_cycle_back(self, *ops):
        '''Contract all to rho as initial, optimize H side'''
        ops = list(ops)
        rho = self.contract_op(self.indexes, self.rho)

        for mid, r in zip(self.indexes[::-1], [None]+self.indexes[1:][::-1]):
            # March to new U
            if r:
                for i in range(len(ops)):
                    ops[i] = self.contract_op((r,), ops[i], hc=True)
            # Contract
            V = [self.contract_hole(H, rho, mid, middle=False) for H in ops]
            # Retreat
            rho = self.contract_op((mid,), rho, hc=True)
            yield (mid, *V)

    def contract_cycle(self, *ops, back=False):
        if back:
            return self.contract_cycle_back(*ops)
        else:
            return self.contract_cycle_for(*ops)

    #@profile
    def minimizeVarE_cycle(self, E=0, forward=True):
        H2 = self.H2-(2*E)*self.H+E**2*np.eye(*self.H.shape)
        l = []
        if forward:
            for sp, V in self.contract_cycle(H2):
                self[sp], varE = opt.minimize_quadratic_local(V, self[sp])
                l.append(varE)
        else:
            for sp, V in self.contract_cycle(H2, back=True):
                U, varE = opt.minimize_quadratic_local(V, self[sp].T.conj())
                self[sp] = U.T.conj()
                l.append(varE)
        return np.array(l)

    def minimizeVar_cycle(self, forward=True):
        l = []
        if forward:
            for sp, V, V2 in self.contract_cycle(self.H, self.H2):
                self[sp], var = opt.minimize_var_local(V, V2, self[sp])
                l.append(var)
        else:
            for sp, V, V2 in self.contract_cycle(self.H, self.H2, back=True):
                U, var = opt.minimize_var_local(V, V2, self[sp].T.conj())
                self[sp] = U.T.conj()
                l.append(var)
        return np.array(l)

    def minimizeVar(self, n=100, rel=1e-10):
        last = np.inf
        for i in range(n):
            cur = self.minimizeVar_cycle()
            print(i, cur)
            if last-cur < rel*cur:
                break
            last=cur
        return cur

def minimize_local(H, rho, depth=4, L=None, dim=2, n=100, rel=1e-6):
    Y = Layers(rho, H, depth=depth, L=L, dim=dim)
    last = np.inf
    for i in range(n):
        cur = self.minimizeVar_cycle()
        print(i, cur)
        if last-cur < rel*cur:
            break
        last=cur
    return Y.contract_rho_full()


if __name__ == "__main__":
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_XX
    n = 2   # dof = 2**(2n) = 64
    d = 1   # At least 2**(2(n-2))
    arg_tpl = {"n": n, "delta": 0.54, "g": 0.1}
    H = Hamilton_XZ(n)['H']
    #print(H)
    rho = Rho.rho_prod_even(n, n/2, amp=0.1, rs=np.random)
    Y = Layers(rho, H, D=d)
    for i in range(10):
        print(i, Y.minimizeVarE_cycle(forward=True))
