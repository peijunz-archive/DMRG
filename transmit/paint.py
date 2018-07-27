import scipy.linalg as la
import numpy as np
from functools import partial
import bhopping as bh
from scipy.misc import imresize, imsave
from .cat import *
import matplotlib.animation as animation

def evolve_graph(H, wavepack, n=12, kd=1, fname=None, sample=None):
    T = np.pi*H['u']/H['h']**2
    if sample is None:
        sample = linspace(0, T, n+1)
    T2 = T*1.1
    dim = int(2*(np.abs(wavepack['alpha'])**2+1)*kd)
    L = 2*wavepack['center']+int(2*T2*H['g'])+1
    psi = bh.BMPS(dim, L, H, wavepack)
    phi = bh.BMPS(dim, L, H, wavepack)
    td = H['h']**2*10
    tpl = "H[{}]\nwave[{}]\ndim{}_L{}_samples{}".format(H, wavepack, dim, L, td)
    if fname is None:
        pre = hash(tpl)%(2**31-1)
    else:
        pre = fname
    #print(pre, dt, tpl)
    l = []
    #print(int(round(T2/dt))+1)
    dt=0
    for i in range(len(sample)):
        if i>0:
            dt = T*(sample[i]-sample[i-1])
            print(dt*td)
            psi.evolve(dt, max(int(dt*td), 3), enable_tail=True)
            phi.lapse(dt)
        else:
            continue
        fname="line_wigner_{}_{}".format(pre, sample[i])
        tt = "{}_t{}".format(tpl, i*dt)
        le = [psi.measure(i, psi.H0) for i in range(psi.L)]
        l.append(le)
        #def Q(z):
            #phi.alpha(z, linear=False)
            #return abs(psi.dot(phi))**2
        print("{:2d}/ Evolved {:.3f}".format(i, dt))
        draw_q_function(psi.wigner, dx=0.4, dy=0.4, fname=fname, tt=tt)#, vmin=0, vmax=1)
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
    x = np.linspace(-4, 4, 81)
    y = np.linspace(-4, 4, 81)

    z = zgrid(x, y, partial(wigner, psi=cat))
    draw_function(x, y, z, fname='test-wigner', title="Wigner func for |2❭+{}|-2❭".format(k))

    z = zgrid(x, y, partial(husimi, psi=cat))
    draw_function(x, y, z, fname='test-husimi', title="Wigner func for |2❭+{}|-2❭".format(k))

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
    L = 2*wavepack['center']+int(2*T2/2*H['g'])+1
    #L = 20
    dim = 10
    #L=10
    psi = bh.BMPS(dim, L, H, wavepack)
    print(T, L)
    psi.evolve(T*1/2, int(T*10), enable_tail=True)
    x = np.linspace(-4, 4, 81)
    y = np.linspace(-4, 4, 81)
    print(psi.wigner(2))
    z = zgrid(x, y, psi.wigner)
    draw_function(x, y, z, fname='test_line_wigner_cat')#, vmin=0, vmax=1)

def draw_evolve(f, video="untitled", T=1, frames=240, fps=10, dpi=100):
    x = np.linspace(-4, 4, 81)
    y = np.linspace(-4, 4, 81)
    state = f(0)
    n = state.shape[0]
    W = wigner_grid(x, y, n)
    print("Grid generated")
    z = wigner_eval(W, state)
    fig, ax, im = draw_function(x, y, z, vmin=-1, vmax=1)
    def update_img(k):
        state = f(k*T/frames)
        z = wigner_eval(W, state)
        im.set_data(z.T)
        return im
    ani = animation.FuncAnimation(fig, update_img, frames+1, interval=33)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(video+'.mp4', writer=writer, dpi=100)
    return ani


if __name__ == "__main__":
    #draw_evolve(partial(kerr, alpha=2, n=20))
    #u = 2*np.sqrt(6)#2*np.sqrt(3)
    #h = 1
    #psi = Single(coherent(2, 30), {'u':u, 'h':h}, 20)
    #T = np.pi*u/(2*h**2)
    #draw_evolve(partial(psi.rho, i=1), T=T, frames=60)
    #testHusimiWigner(2, -1j, 20)
    #run_q()
    H = {"omega0":0, "g":1, "u":0, "h":0, 'K':1}
    wavepack = {"dk":0.5, "center":5, "k_c":-np.pi/2, "trun":False, 'alpha':2}
    #psi = bh.BMPS(50, 1, H, wavepack)
    #l = psi.evolve_measure(np.pi, k=10)
    #print(psi.wigner(-2))
    #H={"omega0":0, "g":1, "u":8, "h":1}
    #wavepack={"dk":0.3, "center":5, "k_c":-np.pi/2, "trun":False, 'alpha':2}
    testwigner(H, wavepack)

    #evolve_graph(H, wavepack, n=12, kd=1, fname='test', sample=[0, 0.52625, 0.52675, 0.52725, 0.52775, 0.52825, 0.52875, 0.52925, 0.52975])
    #print(np.dot(coherent(1, 20), coherent(-1, 20))**2, np.exp(-4))
