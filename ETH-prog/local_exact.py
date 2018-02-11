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
        self.visit_all = list(self._visit_all())

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
    def transform_sites(op, U, i, brk=False, off=0):
        if not brk:
            op2 = np.einsum("ij, kjl->kil", U, op.reshape(2**(off + i), 4, -1))
            # return op2.reshape(op.shape)
            return op2
        else:
            op2 = np.einsum("ij, klm->kimlj", U,
                            op.reshape(2**(off + i), 4, -1))
            # return op2.reshape(*op.shape, 4, 4)
            return op2

    def contract_rho(self):
        rho = self.rho.copy()
        for ind in self.visit_all:
            #H = transform_sites(H, self[ind], ind[1], brk)
            rho = self.transform_sites(rho, self[ind], ind[1])
            rho = self.transform_sites(rho, self[ind].conj(), ind[1], off=self.L)
        N = 2**self.L
        return rho.reshape(N, N)

    def contract_except(self, H, sp):
        '''Contract all U to rho
        Args:
        H   Hermitian operator to contract
        '''
        rho = self.rho.copy()
        for ind in self.visit_all:
            brk = (ind != sp)
            #H = transform_sites(H, self[ind], ind[1], brk)
            rho = self.transform_sites(rho, self[ind], ind[1], brk)
            rho = self.transform_sites(
                rho, self[ind].conj(), ind[1], brk, off=self.L)
        N = 2**self.L
        H = H.reshape(N, N)
        rho = rho.reshape(N, N, 4**4)
        return np.einsum('ij, jik->k', H, rho).reshape(4, 4, 4, 4)

    #@profile
    def contract_until(self, H, sp):
        '''Contract part of U to rho, part to H
        Faster than former'''
        rho = self.rho.copy()
        for ind in self.visit_all:
            #print(ind, sp)
            if ind < sp:
                rho = self.transform_sites(rho, self[ind], ind[1])
                rho = self.transform_sites(
                    rho, self[ind].conj(), ind[1], off=self.L)
        for ind in self.visit_all[::-1]:
            if ind >= sp:
                H = self.transform_sites(H, self[ind].T.conj(), ind[1])
                H = self.transform_sites(H, self[ind].T, ind[1], off=self.L)
        sh1 = (2**sp[1], 4, 2**(self.L - sp[1] - 2))
        sh2 = (*sh1, *sh1)
        H = H.reshape(sh2)
        rho = rho.reshape(sh2)
        return np.einsum('ijklmn, lonipk->ompj', H, rho)

    def contractV(self, H, sp, until=True):
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
        if until:
            hole = self.contract_until(H, sp)
        else:
            hole = self.contract_except(H, sp)
        V = hole.transpose([1, 0, 2, 3])
        return V

    def minimizeVarE_steps(self, H2):
        for sp in self.visit_all:
            yield sp, self.contractV(H2, sp)

    def minimizeVarE_steps_fast(self, H2):
        for sp in self.visit_all:
            yield sp, self.contractV(H2, sp)

    def minimizeVarE_cycle(self, E=0):
        H2 = self.H2-(2*E)*self.H+E**2*np.eye(*self.H.shape)
        for sp, V in self.minimizeVarE_steps(H2):
            self[sp], varE = opt.minimize_quadratic_local(V, self[sp])
        return varE

    def minimizeVar_steps(self):
        for sp in self.visit_all:
            yield sp, self.contractV(self.H, sp), self.contractV(self.H2, sp)


    def minimizeVar_cycle(self):
        for sp, V, V2 in self.minimizeVar_steps():
            self[sp], var = opt.minimize_var_local(V, V2, self[sp])
        return var

    #@profile
    #def minimizeVar_step(self):
        #for sp in self.visit_all:
            #V = self.contractV(self.H, sp)
            #V2 = self.contractV(self.H2, sp)
            #self[sp], var = opt.minimize_var_local(V, V2, self[sp])
        #return var

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
    return Y.contract_rho()

if __name__ == "__main__":
    from analyse import generate_args
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_XX
    n = 6   # dof = 2**(2n) = 64
    d = 2   # At least 2**(2(n-2))
    arg_tpl = {"n": n, "delta": 0.54, "g": 0.1}
    H = Hamilton_XZ(n)['H']
    #print(H)
    rho = Rho.rho_prod_even(n, 3.33333333333333333)
    #H, rho = np.load('H.npy'), 
    #rho = np.load('rho.npy')
    #print(la.eigh(rho))
    minimize_local(H, rho, d, n)
    #Y = Layers(rho, H, depth=d)
    #print(Y.minimizeVar_cycle())
    #print("All Unitaries", Y.visit_all)
    #mins = opt.exact_min_var(H, rho)
    #print(mins)
    #Y.minimizeVar()

    #Em = min_expect(rho, H@H)
    #print(0, Em, Y.contract_all(Y.H2))
    #for i in range(9):
        #print(i+1, Em, Y.minimizeVarE())
