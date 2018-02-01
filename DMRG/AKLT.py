from DMRG import spin
from DMRG.MPS import MPS
import numpy as np
import scipy.linalg as la
from functools import reduce

ss = sum(np.kron(si, si) for si in spin.S(1)[1:])
H = ss + ss @ ss / 3


def AKLT_State(n=30):
    A = 2 * np.array([np.sqrt(2 / 3) * spin.P(),
                      -np.sqrt(1 / 3) * spin.Z() * 2,
                      -np.sqrt(2 / 3) * spin.N()
                      ]).transpose([1, 0, 2])
    M = [A.copy() for i in range(n)]
    M[0] = M[0][:1]
    M[-1] = M[-1][:, :, 1:]
    s = MPS(M, 3)
    s.canon()
    return s


def Energy(s):
    return sum(s.measure(i, H) for i in range(s.L - 1)).real


sz1 = spin.Z(1)


def zz_op(k, i):
    return ((k, sz1), (i, sz1))


def string_op(l, r):
    o = la.expm(np.pi * 1j * sz1)
    return ((l, sz1), *((i, o) for i in range(l + 1, r)), (r, sz1))

def oneop(ops):
    site = ops[0][0]
    opl = [o[1] for o in ops]
    op = reduce(np.kron, opl)
    return site, op

def evolve(s, time=0.1, n=5, k=40):
    p = s.copy()
    l=[]
    for i in range(n):
        p.iTEBD_double(H, time, k)
        s.canon()
        print('Time {:.3f}, Overlap {:.5f}*exp({:.5f}j)'.format(
            (i + 1) * time, np.abs(s.dot(p)), np.angle(s.dot(p))))
        l.append(s.dot(p))
    return l

def Z(s):
    return np.array([s.corr((i, spin.Z(1))) for i in range(s.L)]).real

def Ztj(s, t, n=50):
    k = (s.L - 1) // 2
    delta=np.diff([0,*t])
    l=[]
    for d in delta:
        s.iTEBD_double(H, d, n)
        l.append(Z(s))
    l=np.array(l).real
    return l

def correlator(s):
    for i in range(0, s.L-2):
        zcorr = np.array([s.corr(*zz_op(i, j)) for j in range(i + 1, N)]).real
        print('Z correlation starting at {}:\n{}\n>>> Ratio: {}'.format(i, zcorr, zcorr[1:] / zcorr[:-1]))
        string = np.array([s.corr(*string_op(i, j)) for j in range(i + 1, N)]).real
        print('String Corr: \n{}'.format(string))
        for j in range(i+1, min(i+6,N-1)):
            l = string_op(i, j)
            assert np.abs(s.corr(*l)-s.measure(*oneop(l)))<1e-13

if __name__ == "__main__":
    N=19
    s = AKLT_State(N)
    #s.update_single(spin.P(1), (N-1)//2)
    s.canon()
    l=evolve(s, n=N*2, time=1, k=20)
    print(l)
