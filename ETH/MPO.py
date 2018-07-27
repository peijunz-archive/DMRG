import numpy as np
from functools import reduce


def dense_form(MPO, n):
    start = np.zeros(MPO[0])
    start[0] = 1

    def add(l, m):
        L = len(l)
        return [np.sum([np.kron(l[j], m[j, i]) for j in range(i+1)]) for i in range(L)]
    return reduce(add, [start, *MPO])
