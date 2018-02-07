import numpy as np
import scipy.linalg as la
import ETH.optimization as opt
from ETH.optimization_test import min_expect


class Layers:
    def __init__(self, L, rho, H, depth=4):
        self.D = depth
        self.L = L
        self.d = (self.D + 1) // 2
        self.l = L - 1
        self.rho = rho
        self.H = H
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

    def contract_all(self):
        rho = self.rho.copy()
        for ind in self.visit_all:
            #H = transform_sites(H, self[ind], ind[1], brk)
            rho = self.transform_sites(rho, self[ind], ind[1])
            rho = self.transform_sites(rho, self[ind].conj(), ind[1], off=self.L)
        N = 2**self.L
        H = self.H.reshape(N, N)
        rho = rho.reshape(N, N)
        return np.einsum('ij, ji', H, rho).real

    def contract_except(self, sp):
        '''Contract all U to rho'''
        H = self.H.copy()
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

    def contract_until(self, sp):
        '''Contract part of U to rho, part to H
        Faster than former'''
        H = self.H.copy()
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

    def contractV(self, sp, until=True):
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
            hole = self.contract_until(sp)
        else:
            hole = self.contract_except(sp)
        V = hole.transpose([1, 0, 2, 3])
        return V

    def update(self):
        for sp in self.visit_all:
            V = self.contractV(sp)
            self[sp] = opt.optimize_quadratic(V, self[sp])
        return self.contract_all()
        #sp = self.visit_all[0]
        #V = self.contractV(sp)
        #self[sp] = opt.optimize_quadratic(V, self[sp])


if __name__ == "__main__":
    from ETH import Rho
    from DMRG.Ising import Hamilton_XZ, Hamilton_XX
    n = 3   # dof = 2**(2n) = 64
    d = 4   # At least 2**(2(n-2))
    H = Hamilton_XZ(n)['H']
    H2 = H@H
    # print(H)
    rho = Rho.rho_prod_even(n, 0.6*n)
    print(rho.shape, H.shape)
    Y = Layers(n, rho, H2, depth=d)
    print(Y.visit_all)
    Em = min_expect(rho, H2)
    for i in range(1000):
        print(i, Em, Y.update())
