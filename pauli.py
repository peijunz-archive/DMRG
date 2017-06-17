'''Pauli Matrices

s_0 and s_i are defined, sigma[i] is convenient for indexing

'''
import numpy as np
s0 = np.array([[1, 0], [0, 1]], dtype='double')
sx = np.array([[0, 1], [1, 0]], dtype='double')
sy = np.array([[0, -1j], [1j, 0]], dtype='complex')
sz = np.array([[1, 0], [0, -1]], dtype='double')
sigma = [s0, sx, sy, sz]
