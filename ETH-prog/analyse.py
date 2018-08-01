# -*- coding: utf-8 -*-
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
import argparse
#import cv2

def unified_optimizer(H, rho, D=0, n=100, rel=1e-6):
    if D > 0:
        return ol.minimize_local(H, rho, D=D, dim=2, n=n, rel=rel)
    else:
        return opt.minimize_var_nfix(H, rho, meps=10, nit=n, err=rel)


def uniform2(v, n, rs=np.random):
    if isinstance(v, tuple):
        l = v[0] + v[1] * rs.uniform(-1, 1, n)
        return l
    else:
        return v


def argv_str(v):
    if isinstance(v, tuple):
        return "{}+-{}".format(*v)
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


def Collect(Hf, arg_tpl, _optimize, arg_opt, rs=np.random, rs_rot=np.random, ns=11, nit=10, pre=''):
    n = arg_tpl['n']
    if isinstance(ns, int):
        s = mlinspace(ns)
    else:
        ns, s = len(ns), np.array(ns)
    S = n*s
    R = np.empty([ns, nit, 2**n, 2**n], dtype='complex128')
    H = np.empty_like(R)
    for j in range(nit):
        print("itering", j)
        H4 = Hf(**generate_args(arg_tpl, rs))['H']
        for i, s in enumerate(S):
            print("Entropy S", s)
            print(arg_tpl, s)
            #H4 = Hf(**generate_args(arg_tpl, rs))['H']
            rho = Rho.rho_prod_even(n, s, rs=rs_rot)
            #print(rho, H4)
            print(i, j, H.shape)
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

def plot_diff_rho(diffs, vars, args_H, s, D):
    pass

def plot_varx(mvx, arg_H, s, D):
    bar = imshow(mvx, vmin=0, vmax=1);
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
            print("cv2 not available")
            raise NotImplementedError
            #cv2.imwrite('{}/S={:04.2f}-{:02d}-log.png'.format(path,
            #                                                  s, j), imresize(img, 800, 'nearest'))
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
            #b = Gibbs.rho2beta(H[i, j], R[i, j])
            #grho = Gibbs.beta2rho(H[i, j], b)
            grho = np.eye(*R[i, j].shape)
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

def polarization(rho, L, k=1):
    l = []
    for i in range(L):
        r = rho.reshape((2**i, 2, 2**(L-i-1))*2)
        x = np.einsum("ijkimk, jm", r, sigma[k])
        l.append(x.real)
    return np.array(l)

def plot_scatter_array(x, y):
    L = len(y)
    for i in range(L):
        plot(x[i]*ones_like(y[i]), y[i], 'ko')

def mean_diag(Hf, arg_tpl, _optimize, arg_opt, arg_clt):
    n = arg_tpl['n']
    D = arg_opt['D']
    res = loadData(Hf, arg_tpl, _optimize, arg_opt, arg_clt, pre="_D={}".format(D))
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    dif = np.empty([nit])
    i = 0
    for j in range(nit):
        grho = Gibbs.beta2rho(H[i, j], 0)
        dif[j] = np.mean(np.diag(Rho.compare_all(R[i, j], grho)))
    return dif

def draw_diff_matrix(Hf, arg_tpl, _optimize, arg_opt, arg_clt):
    n = arg_tpl['n']
    D = arg_opt['D']
    res = loadData(Hf, arg_tpl, _optimize, arg_opt, arg_clt, pre="_D={}".format(D))
    H, R, nit, S = res['H'], res['rho'], res['nit'], res['S']
    dif = np.empty([len(S), nit, n, n])
    px = np.empty([len(S), nit, n])
    pz = np.empty([len(S), nit, n])
    E = np.empty([len(S), nit])
    varE = np.empty([len(S), nit])
    for i, s in enumerate(S):
        print('='*50)
        print(">>> Entropy S={}".format(s))
        for j in range(nit):
            E[i, j] = trace2(H[i,j], R[i, j]).real
            b = Gibbs.rho2beta(H[i, j], R[i, j])
            grho = Gibbs.beta2rho(H[i, j], 0)
            varE[i, j] = trace2(H[i,j]@H[i,j], R[i, j]).real-E[i, j]**2
            #print(b, grho)
            dif[i, j] = Rho.compare_all(R[i, j], grho)
            #dif[i, j] = diff_gibbs(R[i, j], H[i, j])
            px[i, j] = polarization(R[i, j], n, 1)**2
            pz[i, j] = polarization(R[i, j], n, 3)**2
            print('-'*50)
            print("Energy", E[i, j])
            print("Variance", varE[i, j])
            #px = polarization(R[i, j], n, 1)
            #py = polarization(R[i, j], n, 2)
            #pz = polarization(R[i, j], n, 3)
            #print((px**2+py**2+pz**2)/2)
            print(np.diag(dif[i, j]))
            #print(b, px[i, j])
            #v = la.eigvalsh(H[i, j])
            #print("bE", b, b*(v[0]-v[-1]).real)
    mdif = mean(dif, axis=1)
    #mvar = var(dif, axis=1, ddof=1)
    mpx = mean(px, axis=1)
    mpz = mean(pz, axis=1)
    mE = mean(E, axis=1)
    mvarE = mean(varE, axis=1)
    #sdif = std(dif, axis=1, ddof=1) / np.sqrt(nit)
    #print(dif[0, 0])
    #plt.close("all")
    #clf()
    #plot_diff_diag(mdif, mvar, arg_tpl, S/n, arg_opt['D'])
    #savefig(fname(Hf, arg_tpl, "figures", "rho-diff-diag.pdf", pre="_D={:02d}".format(D), align=True))
    ##plt.close("all")
    clf()
    plot_diff(diff(mdif[0]), arg_tpl, S/n, arg_opt['D'])
    savefig(fname(Hf, arg_tpl, "figures", "rho-diff.pdf", pre="_D={:02d}".format(D), align=True))
    plt.close("all")
    clf()
    #plot_diff(np.max(dif, axis=1), arg_tpl, S/n, arg_opt['D'])
    #savefig(fname(Hf, arg_tpl, "figures", "rho-diff-max.pdf", pre="_D={:02d}".format(D), align=True))
    #plt.close("all")
    #clf()
    #for i in [0, 1]:
        #plot_diff(dif[:, i], arg_tpl, S/n, arg_opt['D'])
        #savefig(fname(Hf, arg_tpl, "figures", "rho-diff-sample{}.pdf".format(i), pre="_D={:02d}".format(D), align=True))
        #plt.close("all")
        #clf()
    #plot_varx(mpx, arg_tpl, S/n, arg_opt['D'])
    #xlabel("site")
    #ylabel("s")
    #title(r'Polarization of $\sigma_x$ after optimization ($L={n}, J={J}, h={h}, g(\mu,\delta)={g}$, depth={D})'.format(D=D, **arg_tpl))
    #savefig(fname(Hf, arg_tpl, "figures", "var_x.pdf", pre="_D={:02d}".format(D), align=True))

    #clf()
    #plot_varx(mpz, arg_tpl, S/n, arg_opt['D'])
    #xlabel("site")
    #ylabel("s")
    #title(r'Polarization of $\sigma_z$ after optimization ($L={n}, J={J}, h={h}, g(\mu,\delta)={g}$, depth={D})'.format(D=D, **arg_tpl))
    #savefig(fname(Hf, arg_tpl, "figures", "var_z.pdf", pre="_D={:02d}".format(D), align=True))

    #clf()
    #plot_varx(mE, arg_tpl, S/n, arg_opt['D'])
    #xlabel("site")
    #ylabel("s")
    #title(r'Mean energy after optimization (L={n}, J={J}, h={h}, g(\mu,\delta)={g}, depth={D})'.format(D=D, **arg_tpl))
    #savefig(fname(Hf, arg_tpl, "figures", "E.pdf", pre="_D={:02d}".format(D), align=True))


    #clf()
    #errorbar(S/n,
        #mean(varE, axis=1),
        #std(varE, axis=1),
        #capsize=2,
        #label=r"$\mathrm{tr}(\rho H^2)-\mathrm{tr}(\rho H)^2$")
    #grid();
    #legend();
    #xlabel('s')
    #ylabel('Var[E]');
    #savefig(fname(Hf, arg_tpl, "figures", "E_var.pdf", pre="_D={:02d}".format(D), align=True))

    #clf()
    #errorbar(S/n,
        #mean(E, axis=1),
        #std(E, axis=1),
        #capsize=2,
        #label=r"$\mathrm{tr}(\rho H)$")
    #plot_scatter_array(S/n, E)
    #grid();
    #legend();
    #xlabel('s')
    #ylabel('E');
    #savefig(fname(Hf, arg_tpl, "figures", "E.pdf", pre="_D={:02d}".format(D), align=True))

def interactive_run():
    parser = argparse.ArgumentParser(description='Process a batch of jobs')
    parser.add_argument('window', metavar='W', type=float, nargs='+',
                        default=[0.2, 0.5, 1., 2., 4., 8., 16.],
                    help='Window sizes of random field g')
    parser.add_argument('--depth', dest="depth", type=int, nargs='+',
                    default=[0],
                    help='Window sizes of random field g')
    parser.add_argument('--niter', dest='niter',type=int,
                    default=2000,
                    help='Max number to iterate')
    parser.add_argument('--rel', dest='rel', type=float,
                    default=1e-8,
                    help='Max relative error to stop iteration')
    args = parser.parse_args()
    rs = RandomState(16807)
    rs_rot = RandomState(31415926)
    l = [0.2, 0.5, 1, 2, 4, 8, 16]
    nm = []
    for w in args.window:
        for D in args.depth:
        # draw_diff_matrix(
            #Hamilton_TL,
            #{"n":6, "J":1, "h":0.8090, "g":0.945},
            #ol.minimize_local,
            #{'D':D, 'L':6, 'n':2000, 'rel':1e-8},
            #{'rs': rs, 'ns': 5, 'nit': 10},
            #)
            m = mean_diag(
                Hamilton_TL,
                {"n":6, "J": 1, "h":0.809, "g": (0.945, w)},
                unified_optimizer,
                {'D':D, 'n':args.niter, 'rel':args.rel},
                {'rs': rs, 'rs_rot': rs_rot, 'ns': [0.1], 'nit': 15},
                #pre="_D={}".format(D)
                )
            nm.append(np.mean(m))
        print("window {}, D {}, mean |n_i|^2{}".format(w, args.depth, nm))

def ETH_magic_parameter():
    rs = RandomState(16807)
    rs_rot = RandomState(31415926)
    l = []
    for D in [0,2,4,6,8]:
        nm = draw_diff_matrix(
            Hamilton_TL,
            {"n":6, "J": 1, "h":0.809, "g": 0.945},
            unified_optimizer,
            {'D':D, 'n':1000, 'rel':1e-8},
            {'rs': rs, 'rs_rot': rs_rot, 'ns': [0, 0.6/6, 1.2/6], 'nit': 15},
            #pre="_D={}".format(D)
            )
        l.append(np.mean(nm))
    print(l)

if __name__ == "__main__":
    interactive_run()
    #ETH_magic_parameter()
