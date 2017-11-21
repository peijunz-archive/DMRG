import numpy as np
import scipy.linalg as la

from numpy.random import RandomState
from Ising import Hamilton_XZ, Hamilton_XX, Hamilton_TL
from ETH import Rho, Gibbs
from ETH.optimization import *
from pylab import *
from ETH.basic import *
from scipy.misc import imresize
import os

def uniform2(v, n, rs=np.random):
    if isinstance(v, tuple):
        l=v[0]+v[1]*rs.uniform(-1, 1, n)
        print(l)
        return l
    else:
        return v

def argv_str(v):
    if isinstance(v, tuple):
        return "{}±{}".format(*v)
    else:
        return v

def info(Hf, arg_tpl):
    l=["{}={}".format(k, argv_str(v)) for k, v in arg_tpl.items()]
    l.insert(0, Hf.__name__)
    return '_'.join(l)

def fname(Hf, arg_tpl, path="data", post='npy'):
    return "{}/{}.{}".format(path, info(Hf, arg_tpl), post)

def generate_args(arg_tpl, rs=np.random):
    return {k:uniform2(v, arg_tpl['n'], rs) for k, v in arg_tpl.items()}

def Collect(Hf, arg_tpl, arg_opt, rs=np.random, ns=11, nit=10):
    n = arg_tpl['n']
    S = np.linspace(0, n, ns)[:-1]
    R = np.empty([ns, nit, 2**n, 2**n], dtype='complex128')
    H = np.empty_like(R)
    for i, s in enumerate(S):
        print("Entropy S", s)
        for j in range(nit):
            print("itering", j)
            H4=Hf(**generate_args(arg_tpl, rs))['H']
            rho = Rho.rho_prod_even(n, s)#rand_rotate(, np.random)
            H[i, j] = H4
            R[i, j] = minimize_var(H4, rho, **arg_opt)
    result = {'Hamilton':Hf.__name__, 'S': S, 'nit':nit, 'rho':R, 'H':H, **arg_tpl}
    np.save(fname(Hf, arg_tpl), result)
    print(fname(Hf, arg_tpl), 'Data saved!')
    return result

def testConvergence(Hf, arg_tpl, rs=None):
    H=Hf(n, delta, g, rs)
    R = np.empty([nit, 2**n, 2**n], dtype='complex128')
    for i in range(nit):
        rho = rand_rotate(Rho.rho_prod_even(n, s), np.random)
        R[i] = minimize_var(H, rho, nit=10000)
    w, v = la.eigh(H)
    print(w)
    print(la.eigh(rho)[0])
    for i in range(1, nit):
        rho = v.T.conj()@R[i]@v
        print(diag(rho).real)

def testDiagonal(Hf, arg_tpl):
    res = np.load(fname(Hf, arg_tpl)).item()
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    print(arg_tpl)
    for i, s in enumerate(S):
        for j in range(nit):
            w, v = la.eigh(H[i, j])
            rho = v.T.conj()@R[i, j]@v
            k = la.norm([norm(diag(rho, i)) for i in (0,)])/norm(rho)
            print("Entropy {:.4f}, {:.4f}".format(s, k))

def plot_rho(Hf, arg_tpl, zipper=False):
    res = np.load(fname(Hf, arg_tpl)).item()
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    path = info(Hf, arg_tpl)
    try:
        os.mkdir(path)
    except FileExistsError:
        os.system("rm -r {}/*.png".format(path))
    good = np.empty((len(S), nit), 'float')
    for i, s in enumerate(S):
        for j in range(nit):
            w, v = la.eigh(H[i,   j])
            print(w)
            rho = v.T.conj()@R[i, j]@v
            good[i, j] = norm(diag(rho))/norm(rho)
            imsave('{}/S={:04.2f}-{:02d}.png'.format(path, s, j), imresize(abs(rho), 800, 'nearest'))
    if zipper:
        os.system("zip {0}.zip {0}/*.png 1>/dev/null".format(path))
    print(mean(good, axis=1))

def loadData(Hf, arg_tpl, arg_opt=None, arg_clt=None):
    try:
        return np.load(fname(Hf, arg_tpl)).item()
    except FileNotFoundError:
        return Collect(Hf, arg_tpl, arg_opt, **arg_clt)

#def draw_variance(*args, **arg_tpl):
    #res = loadData(*args, **arg_tpl)
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

def draw_rho_e(*args, **arg_tpl):
    res = loadData(*args, **arg_tpl)
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

def draw_diff_rho(Hf, arg_tpl, arg_opt, arg_clt):
    n=arg_tpl['n']
    res = loadData(Hf, arg_tpl, arg_opt, arg_clt)
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
    #print(dif[0, 0])
    cla()
    for i, s in enumerate(S):
        errorbar(arange(-n+1, n), mdif[i], sdif[i], label="S={:.1f}".format(s), capsize=1.5)
    grid()
    title('Diff between rho for {}'.format(info(Hf, arg_tpl)));
    xlabel('Number of traced sites, +/- mean left/right');
    ylabel(r'$|\mathrm{tr}[\rho-\rho_G]|$')
    legend();
    savefig(fname(Hf, arg_tpl, "figures", "rho-diff.pdf"))

if __name__ == "__main__":
    rs=RandomState(164147)
    for w in [0.25, 0.5, 1, 2, 4, 8]:
        draw_diff_rho(Hamilton_XZ,
                      {"n":6, "delta":0.54, "g":(0, w)},
                      {'nit':3000, 'E':5},
                      {'rs':rs, 'ns':10, 'nit':6})
        plot_rho(Hamilton_XZ, {"n":6, "delta":0.54, "g":(0, w)})
    #testConvergence(4, Hamilton_XZ, g=.1, rs=rs)
