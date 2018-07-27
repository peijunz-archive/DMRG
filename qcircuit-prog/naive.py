import numpy as np
from transmit import bhopping as bh
from transmit import cat
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt

def normal_pack(l, k):
    n = len(l)
    N = np.arange(n)
    x = np.average(N, weights=l)
    x2 = np.average(N**2, weights=l)
    sigma = np.sqrt(x2-x**2)
    L = np.exp(-((N-x)/sigma)**2/2)
    L /= np.sum(L)
    L = np.sqrt(L)*np.exp(1j*k*(N-x))
    return L

def fidelity_bound():
    pass

def show_wavepacket():
    N = 2
    A = np.empty([3, 3, N, 48])
    for dk in [0.25]:#, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for i, amp in enumerate([1]):
            for j, c in enumerate([0, -1, 1]): #Coef of K/g
                K = c*amp
                print("Evolve", K, dk)
                H = {"omega0":0, "g":1, "u":0, "h":0, 'K':K}
                wavepack = {"dk":dk, "center":10, "k_c":-np.pi/2, "trun":True, 'alpha':2}
                T = np.pi/amp
                L = 2*wavepack['center']+int(2*T*H['g'])+1
                psi = bh.BMPS(10, L, H, wavepack)
                #wavepack['center'] = 11
                #wavepack['alpha'] = -2j
                phi = bh.BMPS(10, L, H, wavepack)
                L = []
                Theta = np.linspace(-np.pi, np.pi, 49)[:-1]
                k = wavepack['k_c']
                l = [psi.measure(t, psi.N) for t in range(psi.l)]
                L.append(l)
                for m in range(N):
                    print('>>> {}/{}'.format(m, N))
                    psi.evolve(T/N, 60)
                    psi.canon()
                    l = [psi.measure(t, psi.N) for t in range(psi.l)]
                    L.append(l)
                    #ll = []
                    #for theta in Theta:
                        #phi.c_n = 2*normal_pack(l, k)*np.exp(-1j*k*2*np.sin(k)*T+1j*theta)
                        #phi.init_wave()
                        #print(phi.c_n)
                        #print(np.array([phi.measure(i, phi.N) for i in range(phi.l)]))
                        #ll.append(np.abs(psi.dot(phi))**2)
                    #A[i, j, m] = np.array(ll)
                    #print("Measure stage", m)
                    #print(A[i, j, m])
                    #print(A[i, j, m, :24]*A[i, j, m, 24:])
                print(L)
                fig, ax = plt.subplots()
                #im = ax.imshow(L, vmin=0, vmax=max(1, np.max(L)))
                #ax.set_xlabel('X')
                #ax.set_ylabel('t', rotation=0)
                #fig.colorbar(im, orientation='horizontal')
                for i, l in enumerate(L):
                    ax.plot(l, 'o-', label="t={}".format(i*T/N))
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$\langle n\rangle$')
                fig.savefig('wavepacket/K{}_dk{:.2f}.png'.format(K, dk))
    np.save('out.npy', A)
if __name__ == "__main__":
    show_wavepacket()
