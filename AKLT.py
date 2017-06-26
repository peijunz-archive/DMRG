import spin
from MPS import State
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
    M[0] = M[0][1:]
    M[-1] = M[-1][:, :, 1:]
    s = State(M, 3)
    s.canon()
    return s


def Energy(S):
    return sum(s.measure(i, H) for i in range(S.L - 1)).real


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

def evolve(s, time=0.1, n=5):
    p = s.copy()
    for i in range(5):
        p.iTEBD_double(H, time, 40)
        print('Time {:.3f}, Overlap {:.5f}*exp({:.5f}j)'.format(
            (i + 1) * time, np.abs(s.dot(p)), np.angle(s.dot(p))))


if __name__ == "__main__":
    N = 10
    s = AKLT_State(N)
    k = (N - 1) // 2
    E = Energy(s)
    if N < 13:
        print("State\n{}".format(s))
    print('Total Energy is {}, mean energy each bond is {}'.format(E, E / (N - 1)))
    for i in range(0, N-2):
        zcorr = np.array([s.corr(*zz_op(i, j)) for j in range(i + 1, N)]).real
        print('Z correlation starting at {}:\n{}\n>>> Ratio: {}'.format(i, zcorr, zcorr[1:] / zcorr[:-1]))
        string = np.array([s.corr(*string_op(i, j)) for j in range(i + 1, N)]).real
        print('String Corr: \n{}'.format(string))
        for j in range(i+1, min(i+6,N-1)):
            l = string_op(i, j)
            assert np.abs(s.corr(*l)-s.measure(*oneop(l)))<1e-13
    print('Eigen State Time Evolution', '-' * 30)
    evolve(s)
    print('Non-Eigen State Time Evolution', '-' * 30)
    s.M[0][0, 0, 0] = 1
    s.canon()
    evolve(s)
