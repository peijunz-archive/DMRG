import numpy as np


class Layers:
    def __init__(self, W: int, D: int, offset: int=0):
        '''
        Args:
            + D     depth of circuit
            + W     width of circuit, equals to (chain length - 1)
            + offset

        Other Properties:
            + U     Tensor data structure storing merged layers and corresponding local Unitary
                    transformations
            + indices   enumeration of all local unitaries
        '''
        self.D = D
        self.W = W
        self.U = np.empty([self.D, self.W], dtype='object')
        self.offset = offset % 2
        for i, j in self.visit_all():
            self.U[i, j] = np.empty([4, 4], dtype='complex')
        self.indices = list(self.visit_all())

    def __contains__(self, ind):
        return self.U[ind] is not None

    def __getitem__(self, ind):
        '''Get a local Unitary by layer depth and starting x position'''
        return self.U[ind]

    def __setitem__(self, ind, val):
        '''Get a local Unitary by layer depth and starting x position'''
        self.U[ind] = val

    def visit_all(self):
        '''Enumerate all unitaries in order of dependency'''
        for i in range(self.D):
            for j in range(self.W):
                if (i+j+self.offset) % 2 == 0:
                    yield i, j
