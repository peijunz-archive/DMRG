import numpy as np
import scipy.linalg as la

from numpy.random import RandomState
from Ising import Hamilton_XZ
import Rho
import Gibbs
from optimization import *
from pylab import *

def analyse(H, rho):
    var = Rho.energy_var(H, rho)
    E = trace2(rho, H).real
    b = Gibbs.energy2beta(H, E)
    S = -trace2(rho, la.logm(rho)).real/np.log(2)
    var_eth = Gibbs.beta2var(H, b)
    diff_rho = Rho.compare(Gibbs.beta2rho(H, b), rho)
    return (S, b, var, var_eth, *diff_rho)
def fname(n, delta, g, rs):
    fmt = "n=%d_delta=%.2lf_g=%.2lf"
    name = fmt%(n, delta, g)
    if rs is not None:
        name += "random"
        return name
def Measure(n, delta=1/2, g=1/2, ns=11, nit=10, rs=None):
    a = np.empty([ns, nit, 4+2*n-1])
    rs_rho = RandomState(0)
    for i, s in enumerate(np.linspace(0, n, ns)):
        print("Entropy S", s)
        for j in range(nit):
            print("itering")
            H4=Hamilton_XZ(n, delta, g, rs)
            rho = Rho.rotate(Rho.rand_rho_prod(n, rs_rho, s), rs_rho)
            rho = minimize_var(H4, rho, nit=1000)
            a[i, j] = analyse(H4, rho)
    name = fname(n, delta, g, rs)
    np.save("data/"+name+'.npy', a)
    print(name, 'saved!')
    return a
def draw(n, delta, g, ns=6, rs=None):
    name = fname(n, delta, g, rs)
    try:
        a = np.load("data/"+name+'.npy')
    except FileNotFoundError:
        print("Measure", n, delta, g)
        a = Measure(n, delta, g, rs=rs, ns=ns)
    S, b, var, var_eth = a[:, :, :4].transpose([2, 0, 1])
    dif = a[:, :, 4:]
    cla()
    errorbar(mean(S, axis=1),
         mean(var, axis=1),
         std(var, axis=1),
         capsize=2,
         label=r"$\mathrm{tr}(\rho H^2)-\mathrm{tr}(\rho H)^2$")
    grid();
    legend();
    xlabel('S')
    ylabel('Var[E]');
    title('Variance for random H(%d, %.2lf, %.2lf)'%(n, delta, g));
    savefig("figures/"+name+"-Variance.pdf")
    cla()
    ms = mean(S, axis=1)
    m=mean(dif, axis=1)
    ntry = dif.shape[1]
    s=std(dif, axis=1, ddof=1)/np.sqrt(ntry)
    for i in range(ns):
        errorbar(arange(-n+1, n), m[i], s[i], label="{:.1f}".format(abs(ms[i])), capsize=1.5)
    grid()
    title('Diff between rho for random H(%d, %.2lf, %.2lf), n_try=%d'%(n, delta, g, ntry));
    xlabel('Number of traced sites, +/- mean left/right');
    ylabel(r'$|\rho-\rho_G|/|\rho+\rho_G|$')
    legend();
    savefig("figures/"+name+"-rho-diff.pdf")
if __name__ == "__main__":
    rs = RandomState(123581321)
    for i in [0, 0.05, 0.2, 0.5, 1, 2, 4, 8, 16]:
        draw(4, 1./2, i, rs=rs, ns=5)
