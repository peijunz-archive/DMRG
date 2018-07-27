'''Generalized Spin operators and Pauli Matrices

+ S means spin operator,
+ sigma means Pauli matrices for spin 1/2 systemb
'''

import numpy as np
from typing import Tuple


def dof(s):
    '''
    Degree of freedom for spin s space

    Args
    ---
        s: float
            integer or half integer
    Returns
    ---
        int:
            Degree of freedom
    '''
    return round(2 * s + 1)


def I(s=1/2):
    '''Identity Matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            Identity matrix
    '''
    return np.eye(dof(s))


def R(s=1/2):
    '''Reverse all the spins
    Args:
        s   integer or half integer'''
    return I(s)[::-1]


def Z(s=1/2):
    '''S_z spin matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            S_z matrix
    '''
    return np.diag(np.arange(s, -s - 0.1, -1))


def P(s=1 / 2):
    '''S_+ spin matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            S_+ matrix
    '''
    m = np.zeros([dof(s)] * 2)
    for i in range(dof(s) - 1):
        k = s - i
        m[i, i + 1] = np.sqrt(s * (s + 1) - k * (k - 1))
    return m


def N(s=1 / 2):
    '''S_- spin matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            S_- matrix
    '''
    return P(s).transpose()


def X(s=1 / 2):
    '''S_x spin matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            S_x matrix
    '''
    return (P(s) + N(s)) / 2


def Y(s=1 / 2):
    '''S_y spin matrix

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        np.ndarray:
            S_y matrix
    '''
    return (P(s) - N(s)) / (2j)


def S(s=1 / 2):
    '''Spin matrices (S_0, S_x, S_y, S_z)

    Args
    ---
        s:
            integer or half integer
    Returns
    ---
        Tuple[np.ndarray]:
            S_0, S_x, S_y, S_z matrices
    '''
    return I(s), X(s), Y(s), Z(s)


# Pauli σ Matrices
sigma = [I(), 2 * X(), 2 * Y(), 2 * Z()]

if __name__ == '__main__':
    print('Spin 1/2:')
    print(*S(), sep='\n\n')
    print('Spin 1:')
    print(*S(1), sep='\n\n')
    print('Pauli Matrices:')
    print(*sigma, sep='\n\n')
