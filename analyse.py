import numpy as np
import scipy.linalg as la

from numpy.random import RandomState
from Ising import Hamilton_XZ, Hamilton_XX, Hamilton_TL
from ETH import Rho, Gibbs
from ETH.optimization import *
from pylab import *
from ETH.basic import *
def analyse(H, rho):
    var = Rho.energy_var(H, rho)
    E = trace2(rho, H).real
    b = Gibbs.energy2beta(H, E)
    S = -trace2(rho, la.logm(rho)).real/np.log(2)
    var_eth = Gibbs.beta2var(H, b)
    return (S, E, b, var, var_eth)
def fname(n, delta, g, rs):
    fmt = "n=%d_delta=%.2lf_g=%07.4lf"
    name = fmt%(n, delta, g)
    if rs is not None:
        name += "random"
        return name

def Collect(n, Hf, delta=1/2, g=1/2, ns=11, nit=10, rs=None):
    rs_rho = RandomState(0)
    S = np.linspace(0, n, ns)[:-1]
    R = np.empty([ns, nit, 2**n, 2**n], dtype='complex128')
    H = np.empty([ns, nit, 2**n, 2**n], dtype='complex128')
    for i, s in enumerate(S):
        print("Entropy S", s)
        for j in range(nit):
            print("itering", j)
            H4=Hf(n, delta, g, rs)
            H[i, j] = H4
            rho = rand_rotate(Rho.rho_prod_even(n, s), rs_rho)
            R[i, j] = minimize_var(H4, rho, nit=3000)
    result = {'Hamilton':Hf(), 'delta':1/2, 'g':g, 'S': S, 'nit':nit, 'rho':R, 'H':H}
    name = fname(n, delta, g, rs)
    np.save("data/"+name+'.npy', result)
    print(name, 'saved!')
    return result

def testConvergence(n, Hf, delta=1/2, g=1/2, s=2, nit=5, rs=None):
    rs_rho = RandomState(12345)
    H=Hf(n, delta, g, rs)
    R = np.empty([nit, 2**n, 2**n], dtype='complex128')
    for i in range(nit):
        rho = rand_rotate(Rho.rho_prod_even(n, s), rs_rho)
        R[i] = minimize_var(H, rho, nit=10000)
    w, v = la.eigh(H)
    print(w)
    print(la.eigh(rho)[0])
    for i in range(1, nit):
        rho = v.T.conj()@R[i]@v
        print(diag(rho).real)

def loadData(n, Hf, delta, g, ns=6, rs=None):
    name = fname(n, delta, g, rs)
    try:
        return np.load("data/"+name+'.npy').item()
    except FileNotFoundError:
        return Collect(n, Hf, delta, g, rs=rs, ns=ns)

#def draw_variance(*args, **argv):
    #res = loadData(*args, **argv)
    #re
    #S, b, var, var_eth = a[:, :, :4].transpose([2, 0, 1])
    #dif = a[:, :, 4:]
    #cla()
    #errorbar(mean(S, axis=1),
         #mean(var, axis=1),
         #std(var, axis=1),
         #capsize=2,
         #label=r"$\mathrm{tr}(\rho H^2)-\mathrm{tr}(\rho H)^2$")
    #grid();
    #legend();
    #xlabel('S')
    #ylabel('Var[E]');
    #title('Variance for random H(%d, %.2lf, %.2lf)'%(n, delta, g));
    #savefig("figures/"+name+"-Variance.pdf")

def draw_rho_e(*args, **argv):
    res = loadData(*args, **argv)
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    for i, s in enumerate(S):
        for j in range(nit):
            H_ = H[i, j]
            w, v = la.eigh(H_)
            rho = v.T.conj()@R[i, j]@v
            E = trace2(R[i, j], H_).real
            b = Gibbs.energy2beta(H_, E)
            s1 = Gibbs.beta2entropy(H_, b)
            print(la.norm(np.diag(rho).real)/la.norm(rho), s1)

def draw_diff_rho(*args, **argv):
    n, _, delta, g=args
    res = loadData(*args, **argv)
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    dif = np.empty([len(S), nit, 2*n-1])
    for i, s in enumerate(S):
        for j in range(nit):
            b = Gibbs.rho2beta(H[i, j], R[i, j])
            grho = Gibbs.beta2rho(H[i, j], b)
            dif[i, j] = Rho.compare(R[i, j], grho)
            v = la.eigvalsh(H[i, j])
            print("bE", b, b*(v[0]-v[-1]).real/len(v))
    mdif=mean(dif, axis=1)
    sdif=std(dif, axis=1, ddof=1)/np.sqrt(nit)
    name = fname(n, delta, g, rs)
    print(dif[0, 0])
    cla()
    for i, s in enumerate(S):
        errorbar(arange(-n+1, n), mdif[i], sdif[i], label="{:.1f}".format(s), capsize=1.5)
    grid()
    title('Diff between rho for random H(%d, %.2lf, %.2lf), n_try=%d'%(n, delta, g, nit));
    xlabel('Number of traced sites, +/- mean left/right');
    ylabel(r'$|\mathrm{tr}[\rho-\rho_G]|$')
    legend();
    savefig("figures/"+name+res['Hamilton']+"-rho-diff.pdf")
if __name__ == "__main__":
    rs = RandomState(12358121)
    #for i in [0, 0.2, 0.5, 1, 2, 4, 8, 16]:
        #print("Window", i)
    draw_diff_rho(5, Hamilton_TL, 1, 0.9045, rs=rs, ns=8)
    testConvergence(4, Hamilton_XZ, g=.1, rs=rs)
