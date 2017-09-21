'''Used to minimize an operator(matrix) by Matrix Products
M_n=P_nP_{n-1}...P_1M_0
'''

''' Define a mask class, which encloses rho. The add operation between rho and matrix is defined by
rho + h -> exp(ih)@rho@exp(-ih)
'''

import numpy as np
import scipy.linalg as la

#class RhoMatrix:
    #def __init__(self, rho):
        #self.rho = rho
    #def __iadd__(self, h):
        #U = la.expm(1j*h)
        #self.rho = U@rho@U.T.conj()

def trmul(A, B):
    return np.sum(A*B.T)

def B(rho, H):
    A=1j*(rho@H)
    A+=A.T.conj()
    return A

def gradient(H, rho, H2=None):
    if H2 is None:
        H2=H@H
    coef=2*trmul(rho, H)
    return B(rho, H2)-coef*B(rho, H)

def minimize_var(fun, rho, H, eps=0.1):
    H2 = H@H
    for i in range(100):
        h = eps * gradient(H, rho, H2)
        U = la.expm(1j*h)
        rho = U@rho@U.T.conj()
    return rho

