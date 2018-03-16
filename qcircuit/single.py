import cat
import numpy as np
import scipy.linalg as la
class Single:
    def __init__(self, s, H, dim2=10):
        r = np.zeros(dim2)
        r[0] = 1
        self.dim = (len(s), dim2)
        self.state = np.kron(s, r)
        self.u = H['u']
        self.h = H['h']
        self.H = np.zeros([np.prod(self.dim)]*2)
        self.H += self.u*np.kron(np.eye(self.dim[0]), np.diag(np.arange(self.dim[1])))
        self.a = np.diag(np.sqrt(np.arange(1, self.dim[0])), k=1)
        self.a_p = np.diag(np.sqrt(np.arange(1, self.dim[0])), k=-1)
        self.b = np.diag(np.sqrt(np.arange(1, self.dim[1])), k=1)
        self.b_p = np.diag(np.sqrt(np.arange(1, self.dim[1])), k=-1)
        self.H += self.h*np.kron(np.diag(np.arange(self.dim[0])), self.b+self.b_p)

    def evolve(self, t):
        self.state = la.expm(-1j*self.H*t) @ self.state

    @property
    def S(self):
        return self.state.reshape(self.dim)

    def husumi(self, alpha):
        alp = cat.coherent(alpha, self.dim[0])
        #S = self.state.reshape(self.dim)
        return la.norm(np.einsum('i, ij', alp, self.S))

    def wigner(self, alpha):
        aa = alpha*self.a_p
        aa -= aa.T.conj()
        D = la.expm(aa)
        P = (-1)**np.arange(self.dim[0])
        #print(D.shape, P.shape)
        return np.einsum('ij,j,jk,kl,li', D, P, D.T.conj(), self.S, self.S.T.conj()).real


psi = Single(cat.coherent(2, 60), {'u':10, 'h':1}, 10)
psi.evolve(np.pi*10/2)
print(psi.husumi(0))
#cat.draw_q_function(psi.husumi, dx=0.2, dy=0.2, fname="single", tt="single")#, vmin=0, vmax=1)
cat.draw_q_function(psi.wigner, dx=0.1, dy=0.1, fname="single-test", tt="single", grid=True)#, vmin=0, vmax=1)
print(psi.wigner(1))
print(psi.husumi(0))
