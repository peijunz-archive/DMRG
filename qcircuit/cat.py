import scipy.linalg as la
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import bhopping as bh
from scipy.misc import imresize, imsave
from matplotlib.colors import Normalize

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

def factorial(n):
    l=np.arange(n, dtype='float64')
    l[0]=1
    return np.cumprod(l)

def fock(k, N=50):
    a=np.zeros([N])
    a[k]=1
    return a

def coherent(alpha, N=50):
    den = np.sqrt(factorial(N))
    a = (alpha**np.arange(N))/den
    a/=la.norm(a)
    return a

def wigner(alpha, psi):
    N = len(psi)
    a_p=np.diag(np.sqrt(np.arange(1, N)), k=-1)
    aa = alpha*a_p
    aa -= aa.T.conj()
    D = la.expm(aa)
    P = (-1)**np.arange(N)
    phi = D.T.conj()@psi
    w = np.sum(D@P@D.conj().T)
    #if 2*abs(alpha)**2<N:
        #print("Too small N: {}<={}".format(N, 2*abs(alpha)**2))
    nn = np.abs(phi)**2
    return np.dot(nn, P)

def husimi(alpha, psi):
    return abs(np.vdot(coherent(alpha, len(psi)), psi))**2

def draw_q_function(f, xr=(-4,4), yr=(-4,4), dx=0.2, dy=0.2, trun=False, ip="bilinear", cmap="RdBu_r", fname=None, tt=None, vmin=None, vmax=None, grid=True):
    x = np.arange(xr[0], xr[1]+dx/2, dx)
    y = np.arange(yr[0], yr[1]+dy/2, dy)
    xx, yy = np.meshgrid(x, y, sparse=True)
    _f = np.vectorize(f)
    z = _f(xx+1j*yy)
    if trun:
        z=clip(z, 0, None)
    ext = (xr[0]-dx/2, xr[1]+dx/2, yr[0]-dy/2, yr[1]+dy/2)
    cax = plt.imshow(z, interpolation=ip, aspect='equal', extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, norm=CenterNorm())
    #cax = contourf(x,y,z)
    if np.diff(xr)>np.diff(yr):
        plt.colorbar(cax, orientation='horizontal')
    else:
        plt.colorbar(cax)
    if tt:
        plt.title(tt)
    if grid:
        plt.grid(linestyle=':', linewidth=1)
    plt.xticks(np.arange(int(xr[0]), int(xr[1])+0.1, 1.0))
    plt.yticks(np.arange(int(yr[0]), int(yr[1])+0.1, 1.0))
    if fname:
        plt.savefig(fname+'.pdf')
def evolve_graph(H, wavepack, n=12, kd=1):
    T = np.pi*H['u']/H['h']**2
    dt = T/n
    T2 = T*1.1
    dim = int(2*(np.abs(wavepack['alpha'])**2+1)*kd)
    L = 2*wavepack['center']+int(2*T2*H['g'])+1
    psi = bh.BMPS(dim, L, H, wavepack)
    phi = bh.BMPS(dim, L, H, wavepack)
    def Q(z):
        phi.alpha(z, linear=False)
        return abs(psi.dot(phi))**2
    td = H['h']**2*10
    tpl = "H[{}]\nwave[{}]\ndim{}_L{}_samples{}".format(H, wavepack, dim, L, td)
    pre = hash(tpl)%(2**31-1)
    print(pre, dt, tpl)
    l = []
    print(int(round(T2/dt))+1)
    for i in range(int(round(T2/dt))+1):
        if i>0:
            psi.evolve(dt, int(dt*td), enable_tail=True)
            phi.lapse(dt)
        fname="{}_{}".format(pre, i)
        tt = "{}_t{}".format(tpl, i*dt)
        le = [psi.measure(i, psi.H0) for i in range(psi.L)]
        l.append(le)
        print("{:2d} Evolved {:.3f}".format(i, dt))
        draw_q_function(Q, dx=0.4, dy=0.4, fname=fname, tt=tt)#, vmin=0, vmax=1)
        plt.clf()
    l = np.array(l)
    plt.plot(l[:, -1])
    plt.savefig("{}_tail.pdf".format(pre))
    plt.clf()
    l = imresize(l, [600, 600], 'nearest')
    imsave('{}.png'.format(pre), l)
        #title('Linear evolution projected to linear dispersion case');

def testHusimiWigner(alpha=2, k=1, dim=80):
    cat = coherent(alpha, dim)+k*coherent(-alpha, dim)
    cat /= la.norm(cat)
    #f = partial(husimi, psi=cat)
    #draw_q_function(f, fname='husimi', tt="Q func for |2❭+|-2❭", dx=0.1, dy=0.1)
    #plt.clf()
    f = partial(wigner, psi=cat)
    draw_q_function(f, fname='wigner', tt="Wigner func for |2❭+{}|-2❭".format(k), dx=0.1, dy=0.1)

    #plt.clf()
    #cat = coherent(alpha, 30)
    #f = partial(wigner, psi=cat)
    #draw_q_function(f, fname='wigner-single', tt="Wigner func for |2❭", dx=0.1, dy=0.1)

def run_q():
    wavepack = {"dk":0.3, "center":10, "k_c":-np.pi/2, "trun":True, 'alpha':2}
    for u in [10, 20]:
        for h in [1, 1.5, 2, 3, 0.5, 0.2]:
            for kd in [1, 1.5]:
                H = {"omega0":0, "g":1, "u":u, "h":h}
                evolve_graph(H, wavepack, n=24, kd=kd)

def testwigner(H, wavepack, n=12, kd=1):
    T = np.pi*H['u']/H['h']**2
    dt = T/n
    T2 = T*1.1
    dim = int(2*(np.abs(wavepack['alpha'])**2+1)*kd)
    L = 2*wavepack['center']+int(2*T2*H['g'])+1
    L = 2
    dim = 40
    psi = bh.BMPS(dim, L, H, wavepack)
    phi = bh.BMPS(dim, L, H, wavepack)
    def Q(z):
        phi.alpha(z, linear=False)
        return abs(psi.dot(phi))**2
    psi.evolve(T*1/2, int(T*80), enable_tail=True)
    #phi.lapse(T/2*0.95)
    #for i in range(psi.L):
        #print(i, la.norm(psi.M[i]))
    #print(psi.wigner(-2, True))
    draw_q_function(psi.wigner, dx=0.2, dy=0.2, fname="test", tt="test")#, vmin=0, vmax=1)
    plt.clf()
    draw_q_function(Q, dx=0.2, dy=0.2, fname="test2", tt="test")#, vmin=0, vmax=1)

if __name__ == "__main__":
    testHusimiWigner(2, -1j, 60)
    #run_q()
    #testwigner(H={"omega0":0, "g":1, "u":10, "h":1}, 
               #wavepack={"dk":0.3, "center":0, "k_c":-np.pi/2, "trun":False, 'alpha':2})
