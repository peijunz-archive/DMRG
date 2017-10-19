import scipy.optimize as opt
import scipy.linalg as la
import numpy as np
from .basic import *

# Find beta
def rho2beta(H, rho):
    E = trace2(H, rho)
    return energy2beta(H, E)

def energy2beta(H, E):
    '''Find beta=1/kT'''
    return opt.newton(lambda x:beta2energy(H, x)-E, 0).real

# From beta to others
def beta2rho(H, b):
    '''rho for Gibbs state with given beta'''
    R=la.expm(-b*H)
    return R/np.trace(R)

def beta2entropy(H, b):
    '''S for Gibbs state with given beta'''
    R=beta2rho(H, b)
    return entropy(la.eigvalsh(R))

def beta2energy(H, beta):
    '''Find energy'''
    return trace2(H, beta2rho(H, beta)).real

def beta2var(H, x):
    rho = beta2rho(H, x)
    return (trace2(H@H, rho)-trace2(H, rho)**2).real
