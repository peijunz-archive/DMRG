import Rho
import scipy.optimize as opt
import scipy.linalg as la
import ETH
import numpy as np

def beta2rho(H, b):
    '''rho for Gibbs state with given beta'''
    R=la.expm(-b*H)
    return R/np.trace(R)

def beta2energy(H, beta):
    '''Find energy'''
    return ETH.trace2(H, beta2rho(H, beta)).real

def energy2beta(H, E):
    '''Find beta=1/kT'''
    return opt.newton(lambda x:beta2energy(H, x)-E, 0)

def beta2var(H, x):
    return Rho.energy_var(H, beta2rho(H, x))
