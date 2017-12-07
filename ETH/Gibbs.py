import scipy.optimize as opt
import scipy.linalg as la
import numpy as np
from .basic import *

# Find beta
def rho2beta(H, rho):
    E = trace2(H, rho).real
    return energy2beta(H, E)

def energy2beta(H, E, m=10):
    '''Find beta=1/kT'''
    #print(beta2energy(H, m), E, beta2energy(H, -m))
    if beta2energy(H, m)<E<beta2energy(H, -m):
        b = opt.newton(lambda x:beta2energy(H, x)-E, 0, tol=1e-6, maxiter=100).real
        #print("Optimized", b)
        return b
    else:
        #print("UnOptimized", E)
        return m if E<0 else -m

# From beta to others
def beta2rho(H, b):
    '''rho for Gibbs state with given beta'''
    # Rescale the Hamiltonian to give converging expm result
    H_less = H + np.eye(*H.shape)*np.sqrt(la.norm(H))*np.sign(b)
    R=la.expm(-b*H_less)
    return R/np.trace(R)

def beta2entropy(H, b):
    '''S for Gibbs state with given beta'''
    #H_norm = H/np.
    R=beta2rho(H, b)
    return entropy(la.eigvalsh(R))

def beta2energy(H, beta):
    '''Find energy'''
    return trace2(H, beta2rho(H, beta)).real

def beta2var(H, x):
    rho = beta2rho(H, x)
    return (trace2(H@H, rho)-trace2(H, rho)**2).real
