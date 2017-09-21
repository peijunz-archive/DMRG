import scipy as sp
import scipy.linalg as la
import numpy as np
from numpy import diag, trace, linspace, sqrt, tril, triu, zeros
from numpy.random import randn
from scipy.optimize import minimize
from scipy.optimize import newton
from spin import sigma
from functools import reduce, partial
import Ising

class RhoMatrix:
    '''Proxy class'''
    def __init__(self, rho):
        self.rho = rho
    def __iadd__(self, h):
        U = la.expm(1j*h)
        self.rho = U@self.rho@U.T.conj()
        return self
    def __add__(self, h):
        ret = RhoMatrix(self.rho.copy())
        ret += h
        return ret
    def __repr__(self):
        return self.rho.__repr__()
    def __str__(self):
        return self.rho.__str__()

def unity(A):
    '''#DEPRECATED '''
    L=tril(A, -1)
    U=triu(A)
    H=1j*(U+U.T)-(L-L.T)
    return la.expm(H)

def unityx(x):
    '''#DEPRECATED '''
    n=int(sqrt(x.size))
    A=x.reshape(n,n)
    L=tril(A, -1)
    U=triu(A)
    H=1j*(U+U.T)-(L-L.T)
    return la.expm(H)

def trmul(A, B):
    return np.sum(A*B.T)

def _varh(H, rho, H2=None):
    if H2 is None:
        H2=H@H
    res=trmul(H2, rho)-trmul(H, rho)**2
    return res.real

def varh(H, Rho, H2=None):
    return _varh(H, Rho.rho, H2)

def ETH_rho(H, b):
    '''#DEPRECATED '''
    R=la.expm(-b*H)
    return R/trace(R)

def Hamiltonian(n, delta, g):
    '''$H=-\sum (Z_iZ_j-\Delta X_iX_j)-g\sum X_i$'''
    H=-Ising.nearest(n, sigma[1], coef=g)
    H-=Ising.nearest(n, sigma[3], sigma[3])
    H-=delta*Ising.nearest(n, sigma[1], sigma[1])
    return H.todense()

def energy(H, x):
    return trmul(H, ETH_rho(H, x)).real

def energy_var(H, x):
    return varh(H, ETH_rho(H, x))


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


def optimize_varh2(H, rho):
    '''Optimize like iTEBD, 2x2'''
    pass

def beta(H, E):
    return newton(lambda x:energy(H, x)-E, 0)

def product_rho(*L):
    rho=reduce(sp.kron, L)
    return sp.diag(rho/sum(rho))

def rand_rho(H):
    rho = abs(randn(H.shape[0]))
    rho/=sum(rho)
    return sp.diag(rho)

def rand_rho_prod(n):
    rhol = abs(randn(n, 2))+1
    rhol /= np.sum(rhol, axis=1)[:, np.newaxis]
    print(rhol)
    rho = reduce(sp.kron, rhol)
    rho/= sum(rho)
    return sp.diag(rho)

if __name__ == "__main__":
    H4=Hamiltonian(4, 1/2, 1/2)
    rho = rand_rho_prod(4)
    #rho=product_rho([9, 2], [2,4], [6,2])#, [3,4])
    #rho=ETH_rho(H4, 1)+0.1*product_rho([5/7, 2], [2,4], [6,2])#perturbation
    #rho/=trace(rho)
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
