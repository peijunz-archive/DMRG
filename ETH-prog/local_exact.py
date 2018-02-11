import numpy as np
import scipy.linalg as la
import ETH.optimization as opt
from ETH.optimization_test import min_expect


class Layers:
    def __init__(self, rho, H, depth=4, L=None, dim=2):
        '''TODO '''
        self.D = depth
        if L is None:
            self.L = np.int(np.log2(H.size)/np.log2(dim**2))
        else:
            self.L = L
        self.d = (self.D + 1) // 2
        self.l = self.L - 1
        self.rho = rho
        self.H = H
        self.H2 = H@H
        U = np.array([np.eye(4)
                      for i in range(self.d * self.l)], dtype='complex')
        self.U = U.reshape(self.d, self.l, 4, 4)
        self.even_layer = np.arange(0, self.L - 1, 2)
        self.odd_layer = np.arange(1, self.L - 1, 2)
        self.indexes = list(self._visit_all())

    def __getitem__(self, ind):
        layer, pos = ind
        #print(self.U.shape, ind)
        return self.U[layer // 2, pos]

    def __setitem__(self, ind, val):
        layer, pos = ind
        #print(ind)
        self.U[layer // 2, pos] = val

    def _visit_all(self):
        for d in range(self.d):
            # Even layers
            for i in self.even_layer:
                yield (2 * d, i)
            # Odd layers
            if 2 * d + 1 < self.D:
                for i in self.odd_layer:
                    yield (2 * d + 1, i)

    @staticmethod
    def transform_sites(op, U, ind, brk=False, off=0):
        if not brk:
            op2 = np.einsum("ij, kjl->kil", U, op.reshape(2**(off + ind[1]), 4, -1))
            # return op2.reshape(op.shape)
            return op2
        else:
            op2 = np.einsum("ij, klm->kimlj", U,
                            op.reshape(2**(off + ind[1]), 4, -1))
            # return op2.reshape(*op.shape, 4, 4)
            return op2

    def contract_op(self, inds, op, hc=False):
        '''
        Args:
            hc  Hermitian conjugate of U3U2U1, which gives U1+U2+U3+'''
        if not hc:
            for ind in inds:
                op = self.transform_sites(op, self[ind], ind)
                op = self.transform_sites(op, self[ind].conj(), ind, off=self.L)
        else:
            for ind in inds[::-1]:
                op = self.transform_sites(op, self[ind].T.conj(), ind)
                op = self.transform_sites(op, self[ind].T, ind, off=self.L)
        return op

    def contract_rho_full(self):
        rho = self.contract_op(self.indexes, self.rho)
        N = 2**self.L
        return rho.reshape(N, N)

    def contract_hole(self, rho, H, ind, middle=True, hc=True):
        '''
        Aim function
        =hole_{ijkl}U_{ji}U_{lk}^*
        =hole_{ijkl}U_{ji}U_{kl}^+
        =hole_{ijkl}U_{ji}U_{kl}^+
        =V_{ijkl}U_{ij}U_{kl}^+
        ==> V_{ijkl}=hole_{jikl}
        hole_{ijkl} = hole_{kilj}^*
        ==> V_{ijkl} = V_{klji}*
        V_{lkji}=V_{ijkl}*, f=V_{ijkl}U_{ij}U_{kl}^+
        '''
        sh = [2**ind[1], 4, 2**(self.L - ind[1] - 2)]*2
        '''Convention: we are optimizing rho side U, so U at ind is contracted with H'''
        if middle:
            H = self.contract_op((ind,), H, hc=hc)
        return np.einsum('ijklmn, lonipk->mopj', H.reshape(sh), rho.reshape(sh))

    #@profile
    def contract_until(self, H, i):
        '''Contract lower part of U to rho, upper part to H'''
        rho = self.contract_op(self.indexes[:i], self.rho)
        H = self.contract_op(self.indexes[i+1:], H, hc=True)
        return self.contract_hole(rho, H, self.indexes[i])

    #@profile
    def minimizeVarE_steps(self, H2):
        rho = self.rho
        H2 = self.contract_op(self.indexes[1:], H2, hc=True)
        r = self.indexes[0]
        yield r, self.contract_hole(rho, H2, r)

        for l, r in zip(self.indexes[:-1], self.indexes[1:]):
            rho = self.contract_op((l,), rho)
            V=self.contract_hole(rho, H2, r, middle=False)
            H2 = self.contract_op((r,), H2)
            yield r, V

    #@profile
    def minimizeVarE_cycle(self, E=0):
        H2 = self.H2-(2*E)*self.H+E**2*np.eye(*self.H.shape)
        for sp, V in self.minimizeVarE_steps(H2):
            self[sp], varE = opt.minimize_quadratic_local(V, self[sp])
        return varE

    def minimizeVar_steps(self):
        '''Contract all to H as initial, optimize U side'''
        rho = self.rho
        H = self.contract_op(self.indexes[1:], self.H, hc=True)
        H2 = self.contract_op(self.indexes[1:], self.H2, hc=True)
        mid = self.indexes[0]
        yield mid, self.contract_hole(rho, H, mid), self.contract_hole(rho, H2, mid)

        for l, mid in zip(self.indexes[:-1], self.indexes[1:]):
            # March to new U
            rho = self.contract_op((l,), rho)
            # Contract
            V = self.contract_hole(rho, H, mid, middle=False)
            V2 = self.contract_hole(rho, H2, mid, middle=False)
            # Retreat
            H = self.contract_op((mid,), H)
            H2 = self.contract_op((mid,), H2)
            yield mid, V, V2

    def minimizeVar_steps_back(self):
        '''Contract all to rho as initial, optimize H side'''
        rho = self.contract_op(self.indexes[:-1], self.rho)
        H = self.H
        H2 = self.H2
        mid = self.indexes[-1]
        yield mid, self.contract_hole(H, rho, mid, hc=False), self.contract_hole(H2, rho, mid, hc=False)

        for mid, r in zip(self.indexes[:-1][::-1], self.indexes[1:][::-1]):
            # March to new U
            H = self.contract_op((r,), H, hc=True)
            H2 = self.contract_op((r,), H2, hc=True)
            # Contract
            V = self.contract_hole(H, rho, mid, middle=False)
            V2 = self.contract_hole(H2, rho, mid, middle=False)
            # Retreat
            rho = self.contract_op((mid,), rho, hc=True)
            yield mid, V, V2

    def minimizeVar_cycle(self, forward=True):
        if forward:
            steps = self.minimizeVar_steps()
            for sp, V, V2 in steps:
                self[sp], var = opt.minimize_var_local(V, V2, self[sp])
        else:
            steps = self.minimizeVar_steps_back()
            for sp, V, V2 in steps:
                U, var = opt.minimize_var_local(V, V2, self[sp].T.conj())
                print("sub", var)
                self[sp] = U.T.conj()
        return var

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
    Y.minimizeVar(n=n, rel=rel)
    return Y.contract_rho_full()

if __name__ == "__main__":
    from analyse import generate_args
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_XX
    n = 9   # dof = 2**(2n) = 64
    d = 3   # At least 2**(2(n-2))
    arg_tpl = {"n": n, "delta": 0.54, "g": 0.1}
    H = Hamilton_XZ(n)['H']
    #print(H)
    rho = Rho.rho_prod_even(n, 3.33333333333333333)
    #H, rho = np.load('H.npy'), 
    #rho = np.load('rho.npy')
    #print(la.eigh(rho))
    #minimize_local(H, rho, d, n)
    Y = Layers(rho, H, depth=d)
    #Y.minimizeVarE_cycle_fast()
    #print("Second")
    #for sp, V in Y.minimizeVarE_steps(Y.H2):
        #print(sp, V.sum())
    #print("All Unitaries", Y.indexes)
    #mins = opt.exact_min_var(H, rho)
    #print(mins)
    #Y.minimizeVar()

    #Em = min_expect(rho, H@H)
    #print(0, Em, Y.contract_all(Y.H2))
    for i in range(10):
        print(i, Y.minimizeVar_cycle(forward=True))
