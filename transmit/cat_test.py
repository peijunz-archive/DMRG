from . import cat
import pytest
import numpy as np
import scipy.linalg as la

def test_exp_an():
    for n in [1,2,4,8]:
        for r in np.linspace(0, 1, 7):
            for theta in np.linspace(0, 2*np.pi, 13):
                a = cat.a_n(n)
                alpha = r*np.exp(1j*theta)
                m1 = cat.exp_an(alpha, n)
                try:
                    m2 = la.expm(alpha*a)
                    np.testing.assert_allclose(m1, m2)
                except OverflowError:
                    pass

def test_coherent():
    s1 = cat.coherent(1, 20)
    for r in np.linspace(0, 2, 11):
        for theta in np.linspace(0, 2*np.pi, 7):
            z = r*np.exp(1j*theta)
            overlap = cat.husimi(z, s1)
            #overlap = np.abs(np.dot(s1, s2.conj()))**2
            assert abs(overlap-np.exp(-np.abs(z-1)**2))<1e-6, "Q function test fail"

def test_wigner(n=30):
    '''Wigner of alpha state should be gaussian distribution'''
    z0 = 1
    s1 = cat.coherent(z0, n)
    for r in np.linspace(0, 2, 11):
        for theta in np.linspace(0, 2*np.pi, 7):
            z = r*np.exp(1j*theta)
            overlap = cat.wigner(z, s1)
            assert abs(overlap-np.exp(-2*np.abs(z-z0)**2))<1e-6, "Wigner function test fail"

def test_displace(n=30):
    '''Wigner of alpha state should be gaussian distribution'''
    z0 = 1
    s1 = cat.coherent(z0, n)
    for r in np.linspace(0, 2, 11):
        for theta in np.linspace(0, 2*np.pi, 7):
            z = r*np.exp(1j*theta)
            s2 = cat.displace(z-z0, n)@s1
            #assert abs(z-z0)<1.1
            assert abs(cat.husimi(z, s2)-1)<1e-6, "Displacement function test fail"
