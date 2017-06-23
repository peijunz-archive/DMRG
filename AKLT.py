import spin
from MPS import State
import numpy as np

N = 5
A = 2*np.array([np.sqrt(2/3)*spin.sp(),
              -np.sqrt(1/3)*spin.sz()*2,
              -np.sqrt(2/3)*spin.sn()
              ]).transpose([1, 0, 2])
M = [A.copy() for i in range(N)]
M[0] = M[0][:1]
M[-1] = M[-1][:, :, 1:]

ss = sum([np.kron(si, si) for si in spin.sigma(1)[1:]])
ss2 = ss@ss

H = ss + ss2 / 3

s = State(M, 3)
s.canon()
p = s.copy()
for i in range(100):
    s.iTEBD_double(H, 1/100, 100)
    print(np.angle(s.dot(p)), np.abs(s.dot(p)))
print(s.M)
print(s.s)
print(p.s)
