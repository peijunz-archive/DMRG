import scipy as sp
import scipy.linalg as la
import numpy as np
from numpy import diag, trace, linspace, sqrt, tril, triu, zeros
from numpy.random import randn
from scipy.optimize import minimize
from scipy.optimize import newton
from spin import sigma
from functools import reduce, partial
from numpy.random import RandomState
import Ising


def trmul(A, B):
    return np.sum(A*B.T)

def energy_var(H, rho, H2=None):
    '''Energy variance for given H and rho'''
    if H2 is None:
        H2=H@H
    res=trmul(H2, rho)-trmul(H, rho)**2
    return res.real

def Hamiltonian(n, delta, g):
    '''$H=-\sum (Z_iZ_j-\Delta X_iX_j)-g\sum X_i$'''
    H=-Ising.nearest(n, sigma[1], coef=g)
    H-=Ising.nearest(n, sigma[3], sigma[3])
    H-=delta*Ising.nearest(n, sigma[1], sigma[1])
    return H.todense()

# ===========

def _B_Matrix(rho, H):
    '''B=i(rho@H-H@rho)'''
    B=1j*(rho@H)
    B+=B.T.conj()
    return B

def gradient(H, rho, H2=None):
    '''The gradient for rho'=exp(ih)@rho@exp(ih)'''
    if H2 is None:
        H2=H@H
    coef=2*trmul(rho, H)
    return _B_Matrix(rho, H2)-coef*_B_Matrix(rho, H)

def minimize_var(fun, rho, H, eps=0.1):
    H2 = H@H
    for i in range(100):
        print(fun(rho))
        h = eps * gradient(H, rho, H2)
        U = la.expm(1j*h)
        rho = U@rho@U.T.conj()
    return rho

# ============== ETH =====================
def ETH_rho(H, b):
    '''rho for ETH state with given beta'''
    R=la.expm(-b*H)
    return R/trace(R)

def ETH_energy(H, beta):
    '''Find energy'''
    return trmul(H, ETH_rho(H, beta)).real

def ETH_beta(H, E):
    return newton(lambda x:ETH_energy(H, x)-E, 0)

def ETH_energy_var(H, x):
    return energy_var(H, ETH_rho(H, x))


def optimize_varh(H, Rho):
    '''The parameters should be represented by a array of size $n^2$
    before optimized by `minimize`. `unity` with `reshape` will turn
    it into a matrix thus give the variance of energy of $H$. Init
    parameters of optimization is given by zeros which makes $U=I_n$'''
    assert(H.shape==Rho.rho.shape)
    n=H.shape[0]
    def f(x):
        U=unityx(x)
        return varh(H, U@rho@U.T.conj())
    print('Initial Var', f(zeros(n*n)))
    res=minimize(f, zeros(n*n))
    U=unityx(res.x)
    return U@rho@U.T.conj(), res.fun

# ======== rho generation ==============
def product_rho(*L):
    '''Generate rho by product of List of small ones'''
    rho=reduce(sp.kron, L)
    return sp.diag(rho/sum(rho))

def rand_rho(H, rs=None):
    '''Generate huge diagonal rho'''
    if rs is None:
        rs=RandomState(0)
    rho = abs(rs.randn(H.shape[0]))
    rho/=sum(rho)
    return sp.diag(rho)

def rand_rho_prod(n, rs=None):
    if rs is None:
        rs=RandomState(0)
    rhol = abs(rs.randn(n, 2)+1)
    rhol /= np.sum(rhol, axis=1)[:, np.newaxis]
    print(rhol)
    return product_rho(*rhol)

if __name__ == "__main__":
    H4=Hamiltonian(4, 1/2, 1/2)
    rho = rand_rho_prod(4)
    E=trmul(rho, H4).real
    b=beta(H4, E)
    print('Initial\nEnergy = {}, β = {}'.format(E, b))
    var0 = varh(H4, rho)
    #print('Before:\n', rho)
    rho, var1=optimize_varh(H4, rho)
    #print('After minimization:\n', rho)
    E=trmul(rho, H4).real
    b=beta(H4, E)
    print("After minimization\nEnergy = {}, β = {}".format(E, b))
    var2 = energy_var(H4, b)
    print('Init Var: {}, Min Var: {}\nEnergy: {}, ETH Var: {}'.format(var0, var1, E, var2))
