'''
This is for the local version of optimization. The Unitary transformation is restricted to finite local layers. 

State, U, Contraction order, Optimization.

State was already implemented in MPS.py

Decomposition + Cone method

|----------------------------|
| Currently not in todo list |
|----------------------------|

'''

import numpy as np


def site2un(site, layer):
    return (site + (layer % 2)) // 2


def un2site_left(un, layer):
    return (2 * un) + (layer % 2)


def un2site_right(un, layer):
    return (2 * un) + (layer % 2) + 1


def left_un(un, layer):
    if layer % 2:
        return un
    else:
        return un - 1


def right_un(un, layer):
    return leftn + 1


class Layers:
    def __init__(self, L, dim, depth=4):
        self.D = depth
        self.L = L
        self.d = (self.D + 1) // 2
        self.l = L - 1
        self.dim = dim
        U = np.array(eye(dim**2) for i in range(self.d * self.l))
        self.U = U.reshape(self.d, self.l, *[self.dim] * 4)

    def __getitem__(self, ind):
        layer, pos = ind
        return self.U[layer // 2, pos]

    def __setitem__(self, ind, val):
        layer, pos = ind
        self.U[layer // 2, pos] = val

    def lbound(self, ind):
        layer, pos = ind
        root = pos - layer
        if root < 0:
            return 0
        else:
            return root

    def rbound(self, ind):
        layer, pos = ind
        root = pos + layer
        if root > self.L - 2:
            return self.L - 2
        else:
            return root

    def bounds(self, ind):
        return self.lbound(ind), self.rbound(ind)

    def contract(ind, op):
        for i in range(*op):

        pass


class LayeredMPS:
    def __init__(self, state, depth=4):
        '''The unitary transformations are applied to the state,
        layer by layer. In every layer there are two site U. Depth
        of U determined how long the entanglement is preserved. 
        Our rule is that the layer closest to states are 0th layer.
        Consequently, the odd layer is shifted by 1.
        '''
        self.state = state
        self.D = depth
        w = self.state.dim**2

    def

    def left_root(un, layer, 0):
        root = un2site_left(un, layer) - layer
        if root < 0:
            return 0
        else:
            return root

    def right_root(un, layer)
        root = un2site_right(un, layer) + layer
        if root >= self.state.L:
            return self.state.L - 1
        else:
            return root

    def contract()
