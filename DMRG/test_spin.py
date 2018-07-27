'''Test commutation relation'''
from .spin import sigma
import numpy as np


def test_pauli():
    np.testing.assert_allclose(sigma[0], np.eye(2))
    s = sigma[1:]
    '''Test anticommutator'''
    for i in range(3):
        for j in range(3):
            anti = s[i]@s[j]+s[j]@s[i]
            np.testing.assert_allclose(anti, 2*((i == j)*np.eye(2)))
    '''Test commutator'''
    for i in range(3):
        comm = s[i-1]@s[i]-s[i]@s[i-1]
        np.testing.assert_allclose(2j*s[i-2], comm)
