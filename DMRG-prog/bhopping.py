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

def init_state(s0, L):
    s0 = np.array(s0)
    n = len(s0)
    ground = np.zeros([n])
    ground[0] = 1
    M = [ground.reshape([1, n, 1]).copy() for i in range(n)]
    M[0] = s0.reshape([1, n, 1])
    s = State(M, n, trun=20)
    s.canon()
    return s

def evolve(s, H, E, time, n=5, k=100):
    p = s.copy()
    l=[]
    for i in range(n):
        p.iTEBD_double(H, time, k)
        s.canon()
        print('Time {:.3f}, Overlap {:.5f}*exp({:.5f}j)'.format(
            (i + 1) * time, np.abs(s.dot(p)), np.angle(s.dot(p))))
        l.append(np.abs(s.dot(p)))
        le=[p.measure(i, E) for i in range(5)]
        print(le)
    return l

if __name__ == '__main__':
    s = init_state([5, 4, 3, 2, 1], 5)
    evolve(s, H1(5), H0(5), time=1, n=30)
