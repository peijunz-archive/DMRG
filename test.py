import unittest
import numpy as np
import scipy.linalg as la
from MPS import State
from spin import sigma


class TestMPS(unittest.TestCase):
    def testCanon(self):
        '''Test unity of state and circle matrix s after'''
        s = State((1, 2, 2))
        s.M[0][0, :, :] = np.array([[1, 2 + 1j], [9j, 6j]])
        s.M[1][:, :, 0] = np.array([[- 1j, 2 - 3j], [0.3, 4]])
        s.M[1][:, :, 1] = np.array([[5 - 1j, 2 - 3j], [0.3, 4]])
        s.canon()
        self.assertAlmostEqual(s.dot(s), 1, msg='Non-unitary dot product')
        for i in s.s:
            self.assertAlmostEqual(la.norm(i), 1, msg='Non-unitary s matrix')
        for i in range(s.L):
            TAA = np.einsum("ijk, ijn->kn", s.A(i).conj(), s.A(i))
            BBT = np.einsum("ijk, ljk->il", s.B(i), s.B(i).conj())
            self.assertAlmostEqual(la.norm(TAA - np.eye(s.xr[i])), 0,
                                   msg="Left orthonormalization failed")
            self.assertAlmostEqual(la.norm(BBT - np.eye(s.xl[i])), 0,
                                   msg="Right orthonormalization failed")
        return True

    def testTwoBody(self):
        '''For Hamiltonian Z*Z, the eigenvec is ++, +-, -+, --'''
        s = State.naive([1, 1], [1, 1])
        eig_states = [[[0, 1], [0, 1]], [[0, 1], [1, 0]],
                      [[1, 0], [0, 1]], [[1, 0], [1, 0]]]
        eig_vals = np.array([1, -1, -1, 1])
        s.canon()
        n = 20
        U = la.expm(np.pi / n * 1j *
                    np.kron(sigma[3], sigma[3])).reshape([2, 2, 2, 2])
        phi = np.exp(np.pi / n * 1j * eig_vals)
        L0 = np.array([s[eig_states[j]] for j in range(4)])
        H0 = s.corr((0, sigma[3]), (1, sigma[3]))
        #print("Initial", L0)
        for i in range(n):
            s.update_double(U, 0)
            self.assertAlmostEqual(H0, s.corr((0, sigma[3]), (1, sigma[3])))
            L = np.array([s[eig_states[j]] for j in range(4)])
            L0 *= phi
            #print("L0 expected", L0)
            #print("L0 get", L)
            self.assertAlmostEqual(la.norm(L - L0), 0)
        return True

    def testAKLT(self):
        pass


if __name__ == "__main__":
    unittest.main()
