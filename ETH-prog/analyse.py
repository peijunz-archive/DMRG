import numpy as np
import scipy.linalg as la

from numpy.random import RandomState
from DMRG.Ising import Hamilton_XZ, Hamilton_XX, Hamilton_TL
from DMRG.spin import sigma
from ETH import Rho, Gibbs
import ETH.optimization as opt
import ETH.optimize_layers as ol
from pylab import *
from ETH.basic import *
from scipy.misc import imresize
import os
import cv2


def uniform2(v, n, rs=np.random):
    if isinstance(v, tuple):
        l = v[0] + v[1] * rs.uniform(-1, 1, n)
        print(l)
        return l
    else:
        return v


def argv_str(v):
    if isinstance(v, tuple):
        return "{}±{}".format(*v)
    else:
        return v


def info(Hf, arg_tpl, align=False):
    l = ["{}={}".format(k, argv_str(v)) for k, v in arg_tpl.items()]
    l.insert(0, Hf.__name__)
    return '_'.join(l)


def fname(Hf, arg_tpl, path="data", post='npy', pre="", align=False):
    return "{}/{}{}.{}".format(path, info(Hf, arg_tpl, align), pre, post)


def generate_args(arg_tpl, rs=np.random):
    d = {k: uniform2(v, arg_tpl['n'], rs) for k, v in arg_tpl.items()}
    return d


def Collect(Hf, arg_tpl, _optimize, arg_opt, rs=np.random, ns=11, nit=10, pre=''):
    n = arg_tpl['n']
    S = mlinspace(ns)*n
    R = np.empty([ns, nit, 2**n, 2**n], dtype='complex128')
    H = np.empty_like(R)
    for j in range(nit):
        print("itering", j)
        H4 = Hf(**generate_args(arg_tpl, rs))['H']
        for i, s in enumerate(S):
            print("Entropy S", s)
            print(arg_tpl, s)
            #H4 = Hf(**generate_args(arg_tpl, rs))['H']
            rho = Rho.rho_prod_even(n, s, rs=np.random)
            #print(rho, H4)
            H[i, j] = H4
            R[i, j] = _optimize(H4, rho, **arg_opt)
    result = {'Hamilton': Hf.__name__, 'S': S,
              'nit': nit, 'rho': R, 'H': H, **arg_tpl}
    np.save(fname(Hf, arg_tpl, pre=pre), result)
    print(fname(Hf, arg_tpl, pre=pre), 'Data saved!')
    return result



def diff_gibbs(rho, H):
    b = Gibbs.rho2beta(H, rho)
    #print(b)
    grho = Gibbs.beta2rho(H, b)
    #l=Rho.compare(rho, grho)
    m=Rho.compare_all(rho, grho)
    return m

def plot_diff(diffs, args_H, s, D, vmax=1):
    w = len(diffs)
    fig, ax = plt.subplots(1, w, figsize=(w*2+1, 3), sharey=True);
    plt.subplots_adjust(top=0.8)
    for i in range(w):
        bar = ax[i].imshow(diffs[i], vmin=0, vmax=vmax, cmap="Reds");
        ax[i].set_title('S/L={:5.3f}'.format(s[i]))
    #plt.subplots_adjust(wspace=0.1)
    fig.colorbar(bar, ax=ax.ravel().tolist(), orientation='horizontal', aspect=50);
    fig.suptitle('Difference between density matrix of Local optimization and Gibbs ensemble. (L={n}, J={J}, h={h}, g(μ, Δ)={g}, depth={D})'.format(D=D, **args_H))
    info = "L{n}_J{J}_h{h}_g{g}".format(**args_H)
    #plt.savefig(info+".pdf")

def plot_varx(mvx, arg_H, s, D):
    bar = imshow(mvx, vmin=0, vmax=1, cmap="Reds");
    colorbar(bar, orientation='horizontal', aspect=50);

def testConvergence(Hf, arg_tpl, rs=None):
    H = Hf(n, delta, g, rs)
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
    res = np.load(fname(Hf, arg_tpl, pre=pre)).item()
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    print(arg_tpl)
    for i, s in enumerate(S):
        for j in range(nit):
            w, v = la.eigh(H[i, j])
            rho = v.T.conj()@R[i, j]@v
            k = la.norm([norm(diag(rho, i)) for i in (0,)]) / norm(rho)
            print("Entropy {:.4f}, {:.4f}".format(s, k))


def plot_rho(Hf, arg_tpl, zipper=False):
    res = np.load(fname(Hf, arg_tpl, pre=pre)).item()
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    path = 'data/' + info(Hf, arg_tpl)
    try:
        os.mkdir(path)
    except FileExistsError:
        os.system("rm -r {}/*.png".format(path))
    good = np.empty((len(S), nit), 'float')
    for i, s in enumerate(S):
        for j in range(nit):
            w, v = la.eigh(H[i,   j])
            rho = v.T.conj()@R[i, j]@v
            good[i, j] = norm(diag(rho)) / norm(rho)
            img = abs(rho)
            imsave('{}/S={:04.2f}-{:02d}-plain.png'.format(path, s, j),
                   imresize(img, 800, 'nearest'))
            img = (np.clip(np.log(np.clip(img / img.max(), 1e-25, None)
                                  ) + 10, 0, None) * 25).astype('uint8')
            cv2.imwrite('{}/S={:04.2f}-{:02d}-log.png'.format(path,
                                                              s, j), imresize(img, 800, 'nearest'))
    if zipper:
        os.system("zip {0}.zip {0}/*.png 1>/dev/null".format(path))
    # print(good)


def loadData(Hf, arg_tpl, _optimize=None, arg_opt=None, arg_clt=None, pre=''):
    try:
        print(fname(Hf, arg_tpl, pre=pre))
        return np.load(fname(Hf, arg_tpl, pre=pre)).item()
    except FileNotFoundError:
        return Collect(Hf, arg_tpl, _optimize, arg_opt, **arg_clt, pre=pre)

# def draw_variance(*args, **arg_tpl):
    #res = loadData(*args, **arg_tpl)
    # re
    #S, b, var, var_eth = a[:, :, :4].transpose([2, 0, 1])
    #dif = a[:, :, 4:]
    # cla()
    # errorbar(mean(S, axis=1),
        #mean(var, axis=1),
        #std(var, axis=1),
        # capsize=2,
        # label=r"$\mathrm{tr}(\rho H^2)-\mathrm{tr}(\rho H)^2$")
    # grid();
    # legend();
    # xlabel('S')
    # ylabel('Var[E]');
    #title('Variance for random H(%d, %.2lf, %.2lf)'%(n, delta, g));
    # savefig("figures/"+name+"-Variance.pdf")


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
            print(la.norm(np.diag(rho).real) / la.norm(rho), s1)


def draw_diff_rho(Hf, arg_tpl, _optimize, arg_opt, arg_clt):
    n = arg_tpl['n']
    res = loadData(Hf, arg_tpl, _optimize, arg_opt, arg_clt)
    print(res)
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    dif = np.empty([len(S), nit, 2 * n - 1])
    for i, s in enumerate(S):
        for j in range(nit):
            b = Gibbs.rho2beta(H[i, j], R[i, j])
            grho = Gibbs.beta2rho(H[i, j], b)
            #print(b, grho)
            dif[i, j] = Rho.compare(R[i, j], grho)
            v = la.eigvalsh(H[i, j])
            #print("bE", b, b*(v[0]-v[-1]).real)
    mdif = mean(dif, axis=1)
    sdif = std(dif, axis=1, ddof=1) / np.sqrt(nit)
    #print(dif[0, 0])
    cla()
    for i, s in enumerate(S):
        errorbar(arange(-n + 1, n), mdif[i], sdif[i],
                 label="S={:.1f}".format(s), capsize=1.5)
    grid()
    title('Diff between rho for {}'.format(info(Hf, arg_tpl)))
    xlabel('Number of traced sites, +/- mean left/right')
    ylabel(r'$|\mathrm{tr}[\rho-\rho_G]|$')
    legend()
    savefig(fname(Hf, arg_tpl, "figures", "rho-diff.pdf"))
    print('-' * 30)
    print(mdif)
    l = (mdif[:, n // 2 - 1] + mdif[:, -n // 2]) / 2

    return l

def varx(rho, L):
    l = []
    for i in range(L):
        r = rho.reshape((2**i, 2, 2**(L-i-1))*2)
        x = np.einsum("ijkimk, jm", r, sigma[1])
        l.append(x.real)
    return 1-np.array(l)

def draw_diff_matrix(Hf, arg_tpl, _optimize, arg_opt, arg_clt):
    n = arg_tpl['n']
    D = arg_opt['D']
    res = loadData(Hf, arg_tpl, _optimize, arg_opt, arg_clt, pre="_D={}".format(D))
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    dif = np.empty([len(S), nit, n, n])
    vx = np.empty([len(S), nit, n])
    for i, s in enumerate(S):
        for j in range(nit):
            b = Gibbs.rho2beta(H[i, j], R[i, j])
            grho = Gibbs.beta2rho(H[i, j], b)
            #print(b, grho)
            dif[i, j] = diff_gibbs(R[i, j], H[i, j])
            vx[i, j] = varx(R[i, j], n)
            #v = la.eigvalsh(H[i, j])
            #print("bE", b, b*(v[0]-v[-1]).real)
    mdif = mean(dif, axis=1)
    mvx = mean(vx, axis=1)
    #sdif = std(dif, axis=1, ddof=1) / np.sqrt(nit)
    #print(dif[0, 0])
    plt.close("all")
    clf()
    plot_diff(mdif, arg_tpl, S/n, arg_opt['D'])
    savefig(fname(Hf, arg_tpl, "figures", "rho-diff.pdf", pre="_D={:02d}".format(D), align=True))
    plt.close("all")
    clf()
    plot_varx(mvx, arg_tpl, S/n, arg_opt['D'])
    xlabel("site")
    ylabel("s")
    title(r'Variance of $\sigma_x$ after optimization (L={n}, J={J}, h={h}, g(μ, Δ)={g}, depth={D})'.format(D=D, **arg_tpl))
    savefig(fname(Hf, arg_tpl, "figures", "var_x.pdf", pre="_D={:02d}".format(D), align=True))

if __name__ == "__main__":
    rs = RandomState(164147)
    #l = [0.2, 0.5, 1, 2, 4, 8, 16]
    l =  [0.2, 16]#, 0.5, 8, 1, 4, 2]
    for D in [2, 4, 6, 5, 8, 10]:
        #draw_diff_matrix(
            #Hamilton_TL,
            #{"n":6, "J":1, "h":0.8090, "g":0.945},
            #ol.minimize_local,
            #{'D':D, 'L':6, 'n':2000, 'rel':1e-8},
            #{'rs': rs, 'ns': 5, 'nit': 10},
            #)
        for w in l:
            (
                Hamilton_TL,
                {"n": 6, "J": 1, "h":0.2, "g": (0, w)},
                ol.minimize_local,
                {'D':D, 'L':6, 'n':300, 'n':2000, 'rel':1e-8},
                {'rs': rs, 'ns': 5, 'nit': 5},
                )
