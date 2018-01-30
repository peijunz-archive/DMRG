import numpy as np

def optimize_quadratic(T):
    h = np.zeros((4,4))
    return U

class Layers:
    def __init__(self, L, rho, H, depth=4):
        self.D = depth
        self.L = L
        self.d = (self.D + 1)//2
        self.l = L-1
        self.rho = rho
        self.H = H
        U = np.array([eye(4) for i in range(self.d*self.l)])
        self.U = U.reshape(self.d, self.l, 4, 4)
        self.even_layer = np.arange(0, self.L, 2)
        self.odd_layer = np.arange(1, self.L, 2)
        self.visit_all = np.array(list(self._visit_all()))

    def __getitem__(self, ind):
        layer, pos = ind
        return self.U[layer//2, pos]

    def __setitem__(self, ind, val):
        layer, pos = ind
        self.U[layer//2, pos]=val

    def _visit_all(self):
        for d in range(self.d):
            # Even layers
            for i in self.even_layer:
                yield (2*d, i)
            # Odd layers
            if 2 * d + 1 < self.D:
                for i in self.odd_layer:
                    yield (2*d+1, i)

    @staticmethod
    def transform_sites(op, U, i, break=False, off=0):
        if not break:
            op2 = einsum("ij, kjl->kil", U, op.reshape(2**(off+i), 4, -1))
            #return op2.reshape(op.shape)
            return op2
        else:
            op2 = einsum("ij, klm->kimlj", U, op.reshape(2**(off+i), 4, -1))
            #return op2.reshape(*op.shape, 4, 4)
            return op2

    def contract_except(self, sp):
        '''Contract all U to rho'''
        H = self.H.copy()
        rho = self.rho.copy()
        for ind in self.visit_all:
            break = (ind != sp)
            #H = transform_sites(H, self[ind], ind[1], break)
            rho = transform_sites(rho, self[ind], ind[1], break)
            rho = transform_sites(rho, self[ind].conj(), ind[1], break, off=self.L)
        N = 2**self.L
        H = H.reshape(N, N)
        rho = rho.reshape(N, N, 4, 4, 4, 4)
        return np.einsum('ij, jiklmn->kmln', H, rho)

    def contract_except_fast(self, sp):
        '''Contract part of U to rho, part to H
        Faster than former'''
        H = self.H.copy()
        rho = self.rho.copy()
        for ind in self.visit_all:
            if ind < sp:
                rho = transform_sites(rho, self[ind], ind[1])
                rho = transform_sites(rho, self[ind].conj(), ind[1], off=self.L)
        for ind in self.visit_all[::-1]:
            if ind > sp:
                H = transform_sites(H, self[ind], ind[1])
                H = transform_sites(H, self[ind].conj(), ind[1], off=self.L)
        sh1 = (2**i, 4, 2**(N-i-2))
        sh2 = (*sh1, *sh1)
        H = H.reshape(sh2)
        rho = rho.reshape(sh2)
        return np.einsum('ijklmn, lonipk->jmop', H, rho)

def contract_except(H, rho, UL)
