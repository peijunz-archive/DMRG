#!/usr/bin/env python3
'''
iTEBD for free bosons with nearest hopping

Non perturbative Hamiltonian H_0 commutes with other parts. 

H = H_0 + H_i

Time evolution U = exp(iHt)=exp(iH_0 t)*exp(iH_i t)

We apply ST expansion to H_i terms, i.e. split remaining H into even sites H_e and odd sites H_o.
We only need to construct one hopping term and using existing code.
'''

from DMRG.MPS import *
from DMRG.ST import ST1, ST2
import numpy as np
import scipy.linalg as la
from functools import reduce
from scipy.misc import imresize
from pylab import *

@np.vectorize
def dist(t):
    return (t+np.pi)%(2*np.pi)-np.pi

class BMPS(MPS):
    def __init__(self, dim, L, para, pack, trun=10):
        self.dim = dim
        M = [self.zero().reshape([1, dim, 1]) for i in range(L)]
        #M[0] = np.array(s0).reshape([1, dim, 1])
        super().__init__(M, dim, trun=trun)
        self.canon()
        self._init_H(**para)
        self._init_wavepacket(**pack)

    def zero(self):
        ground = np.zeros(self.dim)
        ground[0] = 1
        return ground

    def p(self, i):
        p = np.zeros(self.dim)
        if i>=0:
            p[i] = 1
        return np.diag(p)

    def _init_H(self, omega0, g, u, h):
        self.omega0 = omega0
        self.g = g
        self.u = u
        self.h = h
        self.a = np.diag(np.sqrt(np.arange(1, self.dim)), k=1)
        self.a_p = self.a.T.conj()
        # Resonance Cavity
        self.H0 = omega0 * np.diag(np.arange(self.dim))
        # Tail
        self.tail = True
        self.LL = self.L - self.tail
        self.Ht = u*np.diag(np.arange(self.dim))
        # Tail coupling
        self.O = np.empty((self.dim,)*4)
        for i in range(self.dim):
            for j in range(self.dim):
                self.O[i, j]=self.p(j-i)
        self.Htc = h*(self.a_p+self.a)
        # Interaction Hamiltonian between nearest neighbor
        self.H1 = np.kron(self.a_p, self.a)
        self.H1 += self.H1.T.conj()
        self.H1 *= g

    def Bm_single(self, mt):
        return la.expm(-1j*mt*self.Htc)

    def Bm(self, dt):
        B = np.empty((self.dim,)*3, dtype="complex")
        for i in range(self.dim):
            B[i] = self.Bm_single(i*dt)
        return B[:, np.newaxis, :, :]

    def MPO(self, dt):
        mpo = [self.O for i in range(self.L)]
        mpo[0] = self.O[:1]
        mpo[-1] = self.Bm(dt)
        return mpo

    def update_MPO(self, mpo):
        assert len(mpo) == self.L, "Incompatible MPO"
        for i in range(self.L):
            multi = np.einsum("ijk, lmjo->ilokm", self.M[i], mpo[i])
            sh = multi.shape
            shape = (sh[0]*sh[1], sh[2], sh[3]*sh[4])
            #print(i, sh, shape)
            self.M[i] = multi.reshape(shape)
            self.xl[i] = shape[0]
            self.xr[i] = shape[-1]
        self.canon()

    def evolve(self, t, n=100, enable_tail=False):
        def even_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            #@profile
            def _even_update():
                for i in range(0, self.LL - 1, 2):
                    self.update_double(U, i)
            return _even_update
        def odd_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            #@profile
            def _odd_update():
                for i in range(1, self.LL - 1, 2):
                    self.update_double(U, i)
            return _odd_update
        def single(k):
            U1 = la.expm(-1j * self.H0 * k * t)
            U2 = la.expm(-1j * self.Ht * k * t)
            #@profile
            def _single():
                for i in range(self.LL):
                    self.update_single(U1, i)
                self.update_single(U2, self.LL)
            return _single
        def tail(k):
            mpo = self.MPO(k * t)
            #@profile
            def _tail():
                self.update_MPO(mpo)
            return _tail
        if enable_tail:
            ST2((even_update, odd_update, single, tail), n)
        else:
            ST2((even_update, odd_update, single), n)

    def evolve_measure(self, time, n=5, k=10, tail=True):
        #p = self.copy()
        l=[[self.measure(i, self.H0) for i in range(self.LL)]]
        print(l)
        for i in range(n):
            self.evolve(time, k, tail)
            self.canon()
            le = [self.measure(i, self.H0) for i in range(self.LL)]
            #if le[-1] >= l[-1][-1]:
            l.append(le)
            #else:
                #break
            print("> Progress {:3d}/{:}".format(i, n), end='\r')
        return np.array(l)

    def omega(self, k):
        return self.omega0 + 2*g*np.cos(k)

    def _wavepacket(self, dk, center, k_c):
        dx = np.arange(self.LL)-center
        k = 2*np.pi*np.arange(self.LL)/self.LL
        f_k = np.exp(-(dist(k-k_c)/dk)**2/2)
        N_k = np.exp(1j * dx[:, np.newaxis] * k[np.newaxis, :])
        #print((dx[:, np.newaxis] * k[np.newaxis, :]).transpose())
        #print(N_k.transpose())
        return N_k @ f_k

    def _init_wavepacket(self, dk, center, k_c=np.pi/2, n=1, trun=True):
        c_n = self._wavepacket(dk, center, k_c)
        if trun:
            lb, rb = center-self.LL//2, center+self.LL//2
            print(lb, rb)
            if lb >= 0:
                c_n[:lb+1] = 0
            if rb <= self.LL-1:
                c_n[rb:] = 0
        c_n /= la.norm(c_n)/n
        for i in range(self.LL):
            coh = la.expm(c_n[i]*self.a_p)@self.zero()
            coh /= la.norm(coh)
            self.M[i][0, :, 0] = coh

if __name__ == '__main__':
    H = {"omega0":1, "g":0.1, "u":10, "h":0.1}
    wavepack = {"dk":0.3, "center":10, "k_c":-np.pi/2, "trun":False}
    s = BMPS(6, 61, H, wavepack)
    l = s.evolve_measure(13, 15, tail=False)
    save("untruncated2.npy", l)
    #l = s.evolve(10, 20, False)
    ##l = s.test(1)
    #print(l/l[0,0])
    #print(s.M[0])
    l = imresize(l, [600, 600], 'nearest')
    imsave('test.png', l)
