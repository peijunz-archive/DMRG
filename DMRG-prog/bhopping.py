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

class BMPS(MPS):
    def __init__(self, s0, L, trun=10):
        dim = len(s0)
        ground = np.zeros(dim)
        ground[0] = 1
        M = [ground.reshape([1, dim, 1]).copy() for i in range(L)]
        M[0] = np.array(s0).reshape([1, dim, 1])
        super().__init__(M, dim, trun=trun)
        self.canon()
        self.setH()

    def p(self, i):
        p = np.zeros(self.dim)
        if i>=0:
            p[i] = 1
        return np.diag(p)

    def setH(self, omega=1, g=1, u=1, h=1):
        self.a = np.diag(np.sqrt(np.arange(1, self.dim)), k=1)
        # Resonance Cavity
        self.H0 = omega*np.diag(np.arange(self.dim))
        # Tail
        self.tail = True
        self.Ht = u*np.diag(np.arange(self.dim))
        # Tail coupling
        self.O = np.empty((self.dim,)*4)
        for i in range(self.dim):
            for j in range(self.dim):
                self.O[i, j]=self.p(j-i)
        self.Htc = h*(self.a.T.conj()+self.a)
        # Interaction Hamiltonian between nearest neighbor
        self.H1 = np.kron(self.a.T.conj(), self.a)
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

    def evolve(self, t, n=100, enable_tail=True):
        def even_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            def _even_update():
                for i in range(0, self.L - 1 - self.tail, 2):
                    self.update_double(U, i)
            return _even_update
        def odd_update(k):
            U = la.expm(-1j * self.H1 * k * t).reshape([self.dim] * 4)
            def _odd_update():
                for i in range(1, self.L - 1 - self.tail, 2):
                    self.update_double(U, i)
            return _odd_update
        def single(k):
            U1 = la.expm(-1j * self.H0 * k * t)
            U2 = la.expm(-1j * self.Ht * k * t)
            def _single():
                for i in range(self.L - self.tail):
                    self.update_single(U1, i)
                self.update_single(U2, self.L - self.tail)
            return _single
        def tail(k):
            mpo = self.MPO(k * t)
            def _tail():
                self.update_MPO(mpo)
            return _tail
        if enable_tail:
            ST2((even_update, odd_update, single, tail), n)
        else:
            ST2((even_update, odd_update, single), n)
            
    def test(self, t):
        U1 = la.expm(-t*1j * self.H0)
        U2 = la.expm(-t*1j * self.H1).reshape([self.dim] * 4)
        for i in range(self.L - self.tail):
            self.update_single(U1, i)
        #self.update_single(U2, self.L - self.tail)
        #self.update_single(U1, 0)
        for i in range(0, self.L - 1 - self.tail, 2):
            self.update_double(U2, i)
        #self.update_double(U2, 0)
        for i in range(self.L - self.tail):
            self.update_single(U1, i)
        #self.update_single(U2, self.L - self.tail)
        #self.update_single(U1, 0)
        for i in range(0, self.L - 1 - self.tail, 2):
            self.update_double(U2, i)
        #self.update_double(U2, 0)
        #print(s.M[0])
        #print(s.measure(0, s.H0))
        #print(s.measure(0, s.p(0)))
        #print(s.measure(0, s.p(1)))
        #print(s.measure(0, s.p(2)))
        #print(s.measure(0, s.p(3)))
        #print(s.measure(0, s.p(4)))
        
    def evolve_measure(self, time, n=5, k=1):
        #p = self.copy()
        l=[[self.measure(i, self.H0) for i in range(self.L)]]
        for i in range(n):
            self.evolve(time, k)
            #self.canon()
            le = [self.measure(i, self.H0) for i in range(self.L)]
            #if le[-1] >= l[-1][-1]:
            l.append(le)
            #else:
                #break
            #print("> Progress {:3d}/{:}".format(i, n), end='\r')
        return np.array(l)

if __name__ == '__main__':
    s = BMPS([0, 1j, 0, 0, 2j], 10)
    #s = BMPS([0, 0, 0, 0, 1], 10)
    print(array([s.measure(i, s.H0) for i in range(s.L)]))
    for i in range(1):
        s.evolve(1, 1000, False)
        #self.canon()
        print(array([s.measure(i, s.H0) for i in range(s.L)]))
    #l = s.evolve(10, 20, False)
    ##l = s.test(1)
    print(np.save("rho2", einsum("ijk, ilk->jl", s.M[0], s.M[0].conj())))
    print(*s.Sr)
    print(s.measure(3, s.H0))
    print(s.measure(3, s.p(0)))
    print(s.measure(3, s.p(1)))
    print(s.measure(3, s.p(2)))
    print(s.measure(3, s.p(3)))
    print(s.measure(3, s.p(4)))
    #print(l/l[0,0])
    #print(s.M[0])
    #l = imresize(l, [600, 600], 'nearest')
    #imsave('test.png', l)
    #s.test()
