import numpy as np

class LayersStruct:
    def __init__(self, W, D, offset=0):
        '''
        Data structure:
            Internally, local unitaries are stored in merged layers. Every layer contains
            multiple unitaries. Mergerd layer is a collection of even/odd paired layers.
            Its even/odd element is for even/odd layers, respectively.

            When indexing U, conceputual (layer height, x position) is used. The __getitem__
            will automatically convert it into self.U[layer_height//2]. The merged layer is
            organized in even layer first order.

        Args:
            + D     depth of layers
            + W     width of the circuit, if None, calculated from shape of rho and dim

        Other Properties:
            + d     depth of merged layers.
            + l     number of elements for each merged layer
            + U     Tensor data structure storing merged layers and corresponding local Unitary
                    transformations
            + indexes   enumeration of all local unitaries
        '''
        self.D = D
        self.W = W
        self.L = W + 1
        self._D = (self.D + 1) // 2
        self.U = np.empty([self._D, self.W, 4, 4], dtype='complex')
        self.offset = 0
        self.indexes = list(self._visit_all())

    def __contains__(self, ind):
        return (ind[0] - ind[1] + self.offset)%2 == 0

    def __getitem__(self, ind):
        '''Get a local Unitary by layer depth and starting position'''
        layer, pos = ind
        return self.U[layer // 2, pos]

    def __setitem__(self, ind, val):
        '''Get a local Unitary by layer depth and starting position'''
        layer, pos = ind
        self.U[layer // 2, pos] = val

    def _visit_all(self):
        '''Enumerate all unitaries in order of dependency'''
        for i in range(self.D):
            for j in range((i+self.offset)%2, self.W, 2):
                yield i, j
