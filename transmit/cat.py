'''Basic tools of quantum info visualization of state
Also single state tools
'''
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from functools import partial
from joblib import Memory
memory = Memory(cachedir="cache", verbose=0)

def a_n(n):
    '''Annilation operator'''
    return np.diag(np.sqrt(np.arange(1, n)), k=1)

def a_p(n):
    '''Production operator'''
    return np.diag(np.sqrt(np.arange(1, n)), k=-1)

def N(n):
    '''Number operator'''
    return np.arange(n)

def P(n):
    '''Parity operator'''
    return (-1)**np.arange(n)

def expa_old(alpha, n=20):
    '''exponential of annihilation operator'''
    a = np.sqrt(np.arange(1, n))
    cur = np.ones(n, dtype='complex')
    A = np.eye(n, dtype='complex')
    for i in range(1, n):
        cur = cur[1:]*a[:n-i]*(alpha/i)
        np.fill_diagonal(A[:n-i, i:], cur)
    return A

def exp_an(alpha, n=20):
    '''exponential of annihilation operator'''
    return exp_ap(alpha.conj(), n).T.conj()

def exp_ap(alpha, n=20):
    '''exponential of production operator'''
    a = np.sqrt(np.arange(1, n))
    A = np.eye(n, dtype='complex')
    invD = alpha/(np.arange(1, n)[::-1])
    for i in range(1, n):
        A[i, :i] = A[i-1, :i]*(a[i-1]*invD[n-1-i:])
    return A

def coherent(alpha, n=50):
    '''coherent state'''
    l = np.arange(0, n, dtype='complex')
    l[1:] = alpha/np.sqrt(l[1:])
    l[0] = 1
    l = np.cumprod(l)
    l /= la.norm(l)
    return l

def displace(alpha, n=20):
    '''D(a)|b>->|a+b>'''
    return exp_ap(alpha, n)@exp_an(-np.conj(alpha), n)*np.exp(-np.abs(alpha)**2/2)

def wigner_matrix(alpha, n=20):
    '''D(a)PD(-a)'''
    D = displace(alpha, n)
    return D@(P(n)[:, np.newaxis]*D.T.conj())

@memory.cache
def wigner_grid(x, y, n):
    n1, n2 = len(x), len(y)
    W = np.empty([n1, n2, n, n], dtype='complex')
    print(W.shape)
    for i in range(n1):
        for j in range(n2):
            alpha = x[i]+1j*y[j]
            W[i, j] = wigner_matrix(alpha, n)
    return W

def wigner_eval(W, rho):
    n1, n2 = W.shape[:2]
    z = np.empty(W.shape[:2], dtype='double')
    print(z.shape)
    for i in range(n1):
        for j in range(n2):
            z[i, j] = expect(W[i, j], rho)
    return z

def kerr(k, alpha=2, n=20):
    state = coherent(alpha, n)
    state = np.exp(1j*np.pi*N(n)**2*k)*state
    return state

def fock(k, n=50):
    a=np.zeros([n])
    a[k]=1
    return a

def expect(W, state):
    if len(state.shape) == 1:
        return np.dot(state.conj(), np.dot(W, state)).real
    else:
        return np.sum(W*state.T).real

def wigner(alpha, psi):
    '''Expectation value of D(a)PD(-a)'''
    W = wigner_matrix(alpha, psi.shape[0])
    return expect(W, psi)

def husimi(alpha, psi):
    return abs(np.vdot(coherent(alpha, len(psi)), psi))**2

def imshow_xyz(ax, x, y, z, *args, **argv):
    '''x is x of xoy rhs system, i.e. j index of image
       y is y of xoy rhs system, i.e. -i index of image
       z is function of x, y, i.e. z[i, j]=f(x[i], y[j])
       Image is actual image matrix, I[i, j]=z[-j, i]'''
    assert z.shape == (len(x), len(y)), "Shape unmatched"
    ex = np.mean(np.diff(x))/2
    ey = np.mean(np.diff(y))/2
    argv['extent'] = (x[0]-ex, x[-1]+ex, y[0]-ey, y[-1]+ey)
    return ax.imshow((z.T)[::-1, :], *args, **argv)

def zgrid(x, y, f):
    z = x[:, np.newaxis]+1j*y[np.newaxis, :]
    return np.vectorize(f)(z)

class CenterNorm(Normalize):
    def __init__(self, vc=0, cc=0.5, vmin=None, vmax=None, clip=False):
        '''
        Args:
            vc      value of center
            cc      color of center
        '''
        Normalize.__init__(self, vmin, vmax, clip)
        assert 0< cc < 1, "Central color should be in (0, 1)"
        self.vc = vc
        self.cc = cc
    def __call__(self, value, clip=None):
        dv = np.array([self.vc - self.vmin, self.vmax - self.vc])
        dc = np.array([self.cc, 1 - self.cc])
        k = 1/max(dv/dc)
        return np.ma.masked_array((value-self.vc)*k+self.cc)

def draw_function(x, y, z, fname=None, title=None, grid=True, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    im = imshow_xyz(ax, x, y, z, interpolation="bilinear", aspect='equal', cmap="RdBu_r", vmin=vmin, vmax=vmax, norm=CenterNorm())
    ax.set_xlabel(r'$\mathrm{Re}(\alpha)$')
    ax.set_ylabel(r'$\mathrm{Im}(\alpha)$')

    if abs(x[-1]-x[0])>abs(y[-1]-y[0]):
        fig.colorbar(im, orientation='horizontal')
    else:
        fig.colorbar(im)

    if title:
        ax.set_title(title)
    if grid:
        ax.grid(linestyle=':', linewidth=1)
    plt.tight_layout()
    if fname:
        fig.savefig(fname+'.pdf', dpi=320)
    return fig, ax, im

def exact_evolution(k, t, state):
    v2 = coherent(n/k)
    pass

def fidelity(beta, k, n=10):
    '''TODO: Extend to multi mode case?'''
    N = np.arange(n)
    beta = coherent(beta, n)
    U = np.diag(np.exp(1j*N*np.pi*k**2/2))
    l = []
    for m in range(n):
        alpha = coherent(-k*m, n)
        v = np.dot(alpha.conj(), np.dot(U, alpha))*np.abs(beta[m])**2
        l.append(v)
    return np.abs(np.sum(l))**2

class Single:
    def __init__(self, s, H, dim2=10):
        #r = np.zeros(dim2)
        #r[0] = 1
        r = coherent(1, dim2)
        self.dim = (len(s), dim2)
        self.state = np.kron(s, r)
        self.u = H['u']
        self.h = H['h']
        self.H = np.zeros([np.prod(self.dim)]*2)
        self.H += self.u*np.kron(np.eye(self.dim[0]), np.diag(np.arange(self.dim[1])))
        self.a = a_n(self.dim[0])
        self.a_p = a_p(self.dim[0])
        self.b = a_n(self.dim[1])
        self.b_p = a_p(self.dim[1])
        self.H += self.h*np.kron(np.diag(np.arange(self.dim[0])), self.b+self.b_p)
        self._rho = None
        #np.save("test.npy", la.eigvalsh(self.H))

    def evolve(self, t):
        self.state = la.expm(-1j*self.H*t) @ self.state
        self._rho = None

    @property
    def S(self):
        return self.state.reshape(self.dim)

    #@profile
    def rho(self, t=0, i=0):
        if t == 0:
            if self._rho:
                return self._rho[i]
            S = self.S
            self._rho = S@S.T.conj(), S.T.conj()@S
            return self._rho[i]
        else:
            state = la.expm(-1j*self.H*t) @ self.state
            S = state.reshape(self.dim)
            if i==0:
                return S@S.T.conj()
            else:
                return S.T.conj()@S

    def husimi(self, alpha):
        alp = coherent(alpha, self.dim[0])
        #S = self.state.reshape(self.dim)
        return expect(self.rho, alp)

    def wigner(self, alpha, i=0):
        return expect(wigner_matrix(alpha, self.dim[i]), self.rho(i=i))


def fidelity_graph(alpha, k, density):
    m = np.linspace(0, k, k*density+1)
    y = [fidelity(alpha, x) for x in 2*np.sqrt(m)]
    plt.plot(m, y, ',-', linewidth=0.4)
    plt.xlabel(r"$u^2/4h^2$")
    plt.ylabel(r"$|\langle\psi_0|\psi\rangle|^2$")
    plt.xlim(0, k)
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig('fidelity.pdf')

if __name__ == "__main__":
    u = 6
    h = 1
    psi = Single(coherent(2, 30), {'u':u, 'h':h}, 25)
    psi.evolve(np.pi*u/2/h**2)
    x = np.linspace(-4, 4, 81)
    y = np.linspace(-4, 4, 81)
    z = zgrid(x, y, partial(psi.wigner, i=0))
    draw_function(x, y, z, fname="test-expa", title="single", grid=True, vmin=0, vmax=1)
