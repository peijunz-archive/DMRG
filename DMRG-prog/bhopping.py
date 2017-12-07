'''
iTEBD for free bosons with nearest hopping

Non perturbative Hamiltonian H_0 commutes with other parts. 

H = H_0 + H_i

Time evolution U = exp(iHt)=exp(iH_0 t)*exp(iH_i t)

We apply ST expansion to H_i terms, i.e. split remaining H into even sites H_e and odd sites H_o.
We only need to construct one hopping term and using existing code.
'''

from DMRG.MPS import State
import numpy as np
import scipy.linalg as la
from functools import reduce
from scipy.misc import imresize
from pylab import *

def a(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=1)

def H0(n):
    return np.arange(n)

def H0(n):
    return a(n).T.conj()@a(n)

def H1(n):
    H = 1j*np.kron(a(n).T.conj(), a(n))
    H += H.T.conj()
    return H

def init_state(s0, L, trun=10):
    s0 = np.array(s0)
    n = len(s0)
    ground = np.zeros([n])
    ground[0] = 1
    M = [ground.reshape([1, n, 1]).copy() for i in range(L)]
    M[0] = s0.reshape([1, n, 1])
    s = State(M, n, trun=trun)
    s.canon()
    return s

def evolve(s, H, E, time, n=5, k=20):
    p = s.copy()
    l=[[p.measure(i, E) for i in range(p.L)]]
    for i in range(n):
        p.iTEBD_double(H, time, k)
        s.canon()
        le = [p.measure(i, E) for i in range(p.L)]
        if le[-1] >= l[-1][-1]:
            l.append(le)
        else:
            break
        print("> Progress {:3d}/{:}".format(i, n), end='\r')
    return np.array(l)

if __name__ == '__main__':
    s = init_state([5, 4, 3, 2, 1], 30)
    l = evolve(s, H1(5), H0(5), time=0.2, n=100)
    l = imresize(l, [600, 600], 'nearest')
    imsave('test.png', l)
