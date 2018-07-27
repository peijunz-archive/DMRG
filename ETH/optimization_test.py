import ETH.optimization as opt
from ETH.basic import *
import ETH.Rho as Rho
from DMRG.Ising import Hamilton_XZ
import scipy.linalg as la
import numpy as np

n = 2
H = Hamilton_XZ(n)['H']
H2 = H@H
rho = Rho.rho_prod_even(n, n*0.5)
rho = rand_rotate(rho)
V = np.einsum('jk, li->ijkl', rho, H)
V2 = np.einsum('jk, li->ijkl', rho, H2)


def test_local_optimization():
    U, got = opt.minimize_quadratic_local(V2, nit=200)
    expected = opt.min_expect(rho, H2)
    assert abs(expected - got) < 1e-6


def meta_test_df(df, f, eps):
    M, f1, f2 = df
    U = la.expm(1j * eps * M)
    f1_r = (f(U)-f())/eps
    f1_l = (f()-f(U.T.conj()))/eps
    f1_mean = (f1_r+f1_l)/2
    f2_num = (f1_r-f1_l)/eps
    print(f1_l, f1_r, f1, f2, f2_num)
    assert abs(f1_mean - f1) < eps
    assert abs(f2_num - f2) < 10*np.sqrt(eps)*max(abs(f2), 1)


def test_df_quadratic():
    df = opt.df_quadratic_local(V2)
    f = partial(opt.f_quadratic_local, V2)
    eps = 1e-6
    meta_test_df(df, f, eps)


def test_df_var():
    df = opt.df_var_local(V, V2)
    f = partial(opt.f_var_local, V, V2)
    eps = 1e-6
    meta_test_df(df, f, eps)
