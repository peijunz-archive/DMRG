'''Generalized Pauli Matrices and Spin operators

+ S means spin operator,
+ s means sigma operator
'''
import numpy as np


def dof(S):
    return round(2 * S + 1)


def I(S=1 / 2):
    '''Identity Matrix'''
    return np.eye(dof(S))


def Z(S=1 / 2):
    '''S_z spin matrix'''
    return np.diag(np.arange(S, -S - 0.1, -1))


def P(S=1 / 2):
    '''S_+ spin matrix'''
    m = np.zeros([dof(S)] * 2)
    for i in range(dof(S) - 1):
        k = S - i
        m[i, i + 1] = np.sqrt(S * (S + 1) - k * (k - 1))
    return m


def N(S=1 / 2):
    '''S_- spin matrix'''
    return P(S).transpose()


def X(S=1 / 2):
    '''S_x spin matrix'''
    return (P(S) + N(S)) / 2


def Y(S=1 / 2):
    '''S_y spin matrix'''
    return (P(S) - N(S)) / (2j)


def S(s=1 / 2):
    '''Spin matrices (S_0, S_x, S_y, S_z)
    >>> S()
    '''
    return I(s), X(s), Y(s), Z(s)


if __name__ == '__main__':
    print(*S(), sep='\n\n')
    print()
    print(*S(1), sep='\n\n')
