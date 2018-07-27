import numpy as np


class Layers:
    """Layers of Unitaries

    Attributes
    -----
        D: int
            depth of circuit
        W: int
            width of circuit, equals to (chain length - 1)
        offset: bool
            Parity of circuit determining even/odd(for offset 0/1) indexed Unitaries to use
        U:  np.ndarray
            Matrix data structure storing local Unitary
        indices: List[Tuple[int, int]]
            enumeration of all local unitaries
    """
    def __init__(self, W: int, D: int, offset: int=0):
        '''
        Args:
            D: int
                depth of circuit
            W: int
                width of circuit
            offset: bool
                Parity
        '''
        self.D = D
        self.W = W
        self.U = np.empty([self.D, self.W], dtype='object')
        self.offset = offset % 2
        for i, j in self.traversal():
            self.U[i, j] = np.empty([4, 4], dtype='complex')
        self.indices = list(self.traversal())

    def __contains__(self, ind):
        return self.U[ind] is not None

    def __getitem__(self, ind):
        '''Get a local Unitary by layer depth and starting x position'''
        return self.U[ind]

    def __setitem__(self, ind, val):
        '''Get a local Unitary by layer depth and starting x position'''
        self.U[ind] = val

    def traversal(self):
        '''Enumerate all unitaries in order of dependency

        Yields
        -----
            ind: Tuple[int, int]
                index of Unitary
        '''
        for i in range(self.D):
            for j in range(self.W):
                if (i+j+self.offset) % 2 == 0:
                    yield i, j
