#!/usr/bin/env python3
'''
iTEBD for free bosons with nearest hopping

Non perturbative Hamiltonian H_0 commutes with other parts. 

H = H_0 + H_i

Time evolution U = exp(iHt)=exp(iH_0 t)*exp(iH_i t)

The system is a bosonic chain, with a dangling element at the end to represent global coupling.
During evolutin, it can be chosen to not involve the last term.

We apply ST expansion to H_i terms, i.e. split remaining H into even sites H_e and odd sites H_o.
We only need to construct one hopping term and using existing code.
'''
from DMRG.MPS import *
from DMRG.ST import ST1, ST2
import numpy as np
import scipy.linalg as la
from functools import reduce
from scipy.misc import imresize
from .cat import wigner_matrix, expect, coherent

def printf(fmt, *arg, **argv):
    print(fmt.format(*arg, **argv))

@np.vectorize
def dist_circ(t):
    return (t+np.pi)%(2*np.pi)-np.pi

class BMPS(MPS):
    def __init__(self, dim, l, para, pack):
        self.dim = dim
        M = [self.zero().reshape([1, dim, 1]) for i in range(l+1)]
        #M[0] = np.array(s0).reshape([1, dim, 1])
        super().__init__(M, dim, trun=self.dim)
        self._init_H(**para)
        self.wavepacket(**pack)
        self.canon()

    def zero(self):
        ground = np.zeros(self.dim)
        ground[0] = 1
        return ground

    def p(self, i):
        p = np.zeros(self.dim)
        if i>=0:
            p[i] = 1
        return np.diag(p)

    def _init_H(self, omega0=0, g=1, u=2, h=1, K=0):
        self.omega0 = omega0
        self.K = K
        self.g = g
        self.u = u
        self.h = h
        self.a = np.diag(np.sqrt(np.arange(1, self.dim)), k=1)
        self.a_p = self.a.T.conj()
        # Resonance Cavity
        N = np.arange(self.dim)
        self.N = np.diag(N)
        #self.H0 = np.diag(self.omega0*N - K*N**2)
        # Tail
        self.l = self.L - 1
        self.Ht = np.diag(np.arange(self.dim))
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
            multi = np.einsum("ijk, lmjo->ilokm", self.M[i], mpo[i], optimize=False)
            sh = multi.shape
            shape = (sh[0]*sh[1], sh[2], sh[3]*sh[4])
            #print(i, sh, shape)
            self.M[i] = multi.reshape(shape)
            self.xl[i] = shape[0]
            self.xr[i] = shape[-1]
        self.canon()

    def evolve(self, t, n=100, enable_tail=True):
        def even_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            #@profile
            def _even_update():
                for i in range(0, self.l - 1, 2):
                    self.update_double(U, i)
            return _even_update
        def odd_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            #@profile
            def _odd_update():
                for i in range(1, self.l - 1, 2):
                    self.update_double(U, i)
            return _odd_update
        def single(k):
            #print(self.omega0*self.N-self.K*self.N**2)
            U1 = la.expm(-1j * (self.omega0*self.N-(self.K/2)*self.N**2) * k * t)
            #@profile
            def _single():
                for i in range(self.l):
                    self.update_single(U1, i)
            return _single
        def tail(k):
            U2 = la.expm(-1j * self.u * self.Ht * k * t)
            def _single():
                self.update_single(U2, self.l)
            return _single
        def coupling(k):
            mpo = self.MPO(k * t)
            #@profile
            def _tail():
                self.update_MPO(mpo)
            return _tail
        tasks = ()
        if self.g:
            tasks += (even_update, odd_update)
        if self.omega0 or self.K:
            tasks += (single,)
        if self.u:
            tasks += (tail,)
        if self.h and enable_tail:
            tasks += (coupling,)
        ST2(tasks, n)
        self.time += t
        self.c_n = self.wavepacket_mode(self.time, linear=False)

    def evolve_measure(self, time, n=1, k=10):
        #p = self.copy()
        l=[[self.measure(i, self.N) for i in range(self.l)]]
        #print("Initial", l[0])
        for i in range(n):
            printf("{}/{}", i, n)
            self.evolve(time, k)
            self.canon()
            le = [self.measure(i, self.N) for i in range(self.l)]
            #if le[-1] >= l[-1][-1]:
            l.append(le)
            #else:
                #break
            #print(le)
            #print("> Progress {:3d}/{:}".format(i, n), end='\r')
        return np.array(l)

    def exact_evolve(self, time, n, linear=False):
        '''Naive evolution based on exact solution'''
        l=[]
        for i in range(n+1):
            self.wavepacket(*self.pack, t=time*i, trun=False, linear=linear)
            le = [self.measure(i, self.N) for i in range(self.l)]
            l.append(le)
            #print("> Progress {:3d}/{:}".format(i, n), end='\r')
        return np.array(l)

    def alpha(self, alpha, linear=False):
        self.wavepacket(*self.pack, alpha=alpha, t=self.time, trun=False, linear=linear)

    def husimi(self, alpha, linear=False):
        pass

    def wigner(self, alpha, linear=False, threshold=0):
        alpha_list = alpha*self.c_n/la.norm(self.c_n)
        W = [(i, wigner_matrix(alp, self.dim)) for i, alp in enumerate(alpha_list)]
        print(len(W))
        #print(alpha_list)
        return self.corr(*W)

    def omega(self, k, k0):
        return self.omega0 + 2*self.g*np.cos(k)

    def linear_omega(self, k, k0):
        return self.omega0 + 2*self.g*(np.cos(k0) - np.sin(k0)*dist_circ(k-k0))

    def wavepacket_mode(self, t=0, linear=False):
        dk, center, k_c = self.pack
        dx = np.arange(self.l)-center
        k = 2*np.pi*np.arange(self.l)/self.l
        f_k = np.exp(-(dist_circ(k-k_c)/dk)**2/2)
        #print(f_k/la.norm(f_k))
        kx = dx[:, np.newaxis] * k[np.newaxis, :]
        if linear:
            _omega = self.linear_omega
        else:
            _omega = self.omega
        #print(k, k_c)
        #print("Line", self.linear_omega(k, k_c), self.omega(k, k_c))
        wt = t*_omega(k, k_c)[np.newaxis, :]
        N_k = np.exp(1j * (kx-wt))
        #print((dx[:, np.newaxis] * k[np.newaxis, :]).transpose())
        #print(N_k.transpose())
        return N_k @ f_k

    def init_wave(self):
        for i in range(self.l):
            self.M[i][0, :, 0] = coherent(self.c_n[i], self.dim)

    def wavepacket(self, dk, center, k_c=np.pi/2, alpha=1, t=0, trun=True, linear=False):
        self.pack = (dk, center, k_c)
        self.time = t
        self.c_n = self.wavepacket_mode(t, linear)
        if trun:
            lb, rb = center-self.l//2, center+self.l//2
            #print(lb, rb)
            if lb >= 0:
                self.c_n[:lb+1] = 0
            if rb <= self.l-1:
                self.c_n[rb:] = 0
        self.c_n = self.c_n/la.norm(self.c_n)*alpha
        self.init_wave()
            #print(i, abs(coh))
        #print("Good", [self.measure(i, self.N) for i in range(self.l)])
            #print(i, coh)

if __name__ == '__main__':
    from pylab import *
    H = {"omega0":0, "g":0, "u":0, "h":0, 'K':1}
    wavepack = {"dk":0.5, "center":5, "k_c":-np.pi/2, "trun":True, 'alpha':2}
    psi = BMPS(10, 1, H, wavepack)
    l = psi.evolve_measure(np.pi, k=10)
    print(psi.M)
    save("untruncated2.npy", l*4)
    #l = s.evolve(10, 20, False)
    ##l = s.test(1)
    #print(l/l[0,0])
    #print(s.M[0])
    print(l)
    #for theta in linspace(0, 2*pi, 11):
        #phi.alpha_t(exp(1j*theta), tot_time)
        #print("Angle {:12.4e}\tπQ {:12.5e}".format(theta, abs(psi.dot(phi))**2))

