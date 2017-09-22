import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opt

from spin import sigma
from functools import reduce
from numpy.random import RandomState
from Ising import nearest
'''
# Note Sep. 14
+ Done. Use Random State to seed and generate Random Hamiltonian/States
+ TODO Compare both partial $\rho$ and total $\rho$ of ETH and minimized one
+ Save data
+ Test the routine by eigen state with some random unitary transformation
+ Pade approximation
+ Cache some used quantity like $H^2$
+ Use iterative incremental $U=\prod e^{h_i}$
'''
def unity(A):
    '''#DEPRECATED '''
    L=np.tril(A, -1)
    U=np.triu(A)
    H=1j*(U+U.T)-(L-L.T)
    return la.expm(H)

def unityx(x):
    '''#DEPRECATED '''
    n=int(np.sqrt(x.size))
    A=x.reshape(n,n)
    L=np.tril(A, -1)
    U=np.triu(A)
    H=1j*(U+U.T)-(L-L.T)
    return la.expm(H)

def trmul(A, B):
    return np.sum(A*B.T)

def energy_var(H, rho, H2=None):
    '''Energy variance for given H and rho'''
    if H2 is None:
        H2=H@H
    res=trmul(H2, rho)-trmul(H, rho)**2
    #print("Var", trmul(H2, rho), trmul(H, rho))
    return res.real

def Hamiltonian(n, delta, g):
    '''$H=-\sum (Z_iZ_j-\Delta X_iX_j)-g\sum X_i$'''
    H=-nearest(n, sigma[1], coef=g)
    H-=nearest(n, sigma[3], sigma[3])
    H-=delta*nearest(n, sigma[1], sigma[1])
    return H

# ===========
def commuteh(h1, h2):
    '''[H1, H2]'''
    M=h1@h2
    M-=M.T.conj()
    return M
def gradient(H, rho, H2=None):
    '''The gradient for rho'=exp(ih)@rho@exp(ih)

    $$\frac{df(H)}{dx}=g'(H^2)-2g(H)g'(H)=\tr[M^2]$$

    $$\frac{d^2f(H)}{dx^2}=g''(H^2)-2g'^2(H)-2g(H)g''(H)$$
    '''
    if H2 is None:
        H2=H@H
    gh = trmul(rho, H)
    dh2 = H2-(2*gh)*H
    M = 1j*commuteh(rho, dh2)
    g1h = trmul(M, H)
    mrho = commuteh(M, rho)
    f1 = trmul(M, M)
    f2 = trmul(mrho, commuteh(M, dh2)) - 2*g1h**2
    return M, f1.real, f2.real

# ============== ETH =====================
def ETH_rho(H, b):
    '''rho for ETH state with given beta'''
    R=la.expm(-b*H)
    return R/np.trace(R)

def ETH_energy(H, beta):
    '''Find energy'''
    return trmul(H, ETH_rho(H, beta)).real

def ETH_beta(H, E):
    '''Find beta=1/kT'''
    return opt.newton(lambda x:ETH_energy(H, x)-E, 0)

def ETH_energy_var(H, x):
    return energy_var(H, ETH_rho(H, x))

def minimize_var(H, rho, meps=10, nit=100):
    H2 = H@H
    for i in range(nit):
        #print("Step {}, energy var {}".format(i, energy_var(H, rho, H2)))
        M, f1, f2 = gradient(H, rho, H2)
        if f2 < f1/meps:
            x = meps
        else:
            x = f1/f2 # Choose smaller step if does not decrease
            #print(f1, f2, x, x*np.sqrt(f1))
        h1 = - x * M
        U = la.expm(1j*h1)
        rho = U@rho@U.T.conj()
    #print("Step {}, energy var {}".format(nit, energy_var(H, rho, H2)))
    np.save('rho', rho)
    return rho

# ======== rho generation ==============
def product_rho(*L):
    '''Generate rho by product of List of small ones'''
    rho=reduce(np.kron, L)
    print(rho)
    return np.diag(rho/sum(rho))

def rand_rho(H, rs=None):
    '''Generate huge diagonal rho'''
    if rs is None:
        rs=RandomState(0)
    rho = abs(rs.randn(H.shape[0]))
    rho /= sum(rho)
    return np.diag(rho)

def rand_rho_prod(n, rs=None):
    if rs is None:
        rs=RandomState(0)
    rhol = abs(rs.randn(n, 2)+1)
    rhol /= np.sum(rhol, axis=1)[:, np.newaxis]
    return product_rho(*rhol)
def cmp_rho(rho1, rho2):
    print(la.norm(rho1-rho2)/la.norm(rho1+rho2))
    n=int(np.sqrt(rho1.shape[0]))
    r1 = np.trace(rho1.reshape([n]*4), axis1=1, axis2=3)
    r2 = np.trace(rho2.reshape([n]*4), axis1=1, axis2=3)
    print(la.norm(r1-r2)/la.norm(r1+r2))
if __name__ == "__main__":
    H4=Hamiltonian(4, 1/2, 1/2)
    rho = rand_rho_prod(4)
    minimize_var(H4, rho)
    E=trmul(rho, H4).real
    b=ETH_beta(H4, E)
    print('Initial\nEnergy = {}, β = {}'.format(E, b))
    var0 = energy_var(H4, rho)
    rho0=rho.copy()
    rho = minimize_var(H4, rho, nit=10000)
    var1 = energy_var(H4, rho)
    E=trmul(rho, H4).real
    b=ETH_beta(H4, E)
    print("After minimization\nEnergy = {}, β = {}".format(E, b))
    var2 = ETH_energy_var(H4, b)
    print('Init Var: {}, Min Var: {}\nEnergy: {}, ETH Var: {}'.format(var0, var1, E, var2))
    cmp_rho(ETH_rho(H4, b), rho)
