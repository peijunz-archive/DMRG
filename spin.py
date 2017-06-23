'''Generalized Pauli Matrices
'''
import numpy as np

def dof(s):
    return round(2*s+1)

def s0(s=1/2):
    return np.eye(dof(s))

def sz(s=1/2):
    return np.diag(np.arange(s, -s-0.1, -1))

def sp(s=1/2):
    m=np.zeros([dof(s)]*2)
    for i in range(dof(s)-1):
        k=s-i
        m[i, i+1]=np.sqrt(s*(s+1)-k*(k-1))
    return m

def sn(s=1/2):
    return sp(s).transpose()

def sx(s=1/2):
    return (sp(s)+sn(s))/2

def sy(s=1/2):
    return (sp(s)-sn(s))/(2j)

def sigma(s=1/2):
    return s0(s), sx(s), sy(s), sz(s)

if __name__ == '__main__':
    print(*sigma(1), sep='\n')
