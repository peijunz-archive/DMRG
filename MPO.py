'''Matrix Product Operator

Express the Hamiltonian of the system in terms of matrix product operator.

Operators are implemented for automatic generation:
+ Addition of Hamiltonian terms is `+`
+ ⨂ Direct Product of Hamiltonian terms of nearing sites is `*`

'''
import numpy as np
from ast import literal_eval
import spin

op2 = dict(zip(('s0', 'sx', 'sy', 'sz'), spin.S()))


class MPO:
    '''x是行，y是列, op是算符。默认(0,0)和(n,n)有两个恒等算符'''
    width = 4
    lazy = True

    def __init__(self, name='s0', op=None, dim=2):
        self.x = [1]
        self.y = [0]
        #assert(op.shape==(dim, dim))
        self.op = [op]
        self.name = [name]
        self.dof = 0
        self.dim = dim
        self.N = 1

    def eval_op(self):
        for i in range(self.N):
            if self.op[i] is None:
                self.op[i] = eval(self.name[i], op2).copy()

    def __str__(self):
        M = self.tomatrix_name()
        return '\n'.join(
            [''.join(['{:^{}}'.format(j, MPO.width) for j in i]) for i in M]
        )

    def tomatrix_name(self):
        M = np.zeros([self.dof + 2, self.dof + 2], dtype='U10')
        M[0, 0] = '1'
        M[-1, -1] = '1'
        for i in range(self.dof):
            M[i + 1, i + 1] = '↰'
        for x, y, op in zip(self.x, self.y, self.name):
            if M[x, y] and not (op.startswith('+') or op.startswith('-')):
                M[x, y] += '+' + op
            else:
                M[x, y] += op
        return M

    def tomatrix(self, dtype='complex', names=op2):
        M = np.zeros([self.dof + 2, self.dof + 2,
                      self.dim, self.dim], dtype=dtype)
        M[0, 0] = np.eye(self.dim)
        M[-1, -1] = np.eye(self.dim)
        self.eval_op()
        for x, y, op in zip(self.x, self.y, self.op):
            M[x, y] += op
        return M

    def __repr__(self):
        return self.__str__()

    def __iadd__(self, rhs):
        L = self.dof + rhs.dof
        self.x = [i if i <= self.dof else L + 1 for i in self.x]
        x2 = [i + self.dof for i in rhs.x]
        y2 = [i + self.dof if i else 0 for i in rhs.y]
        self.x += x2
        self.y += y2
        self.dof += rhs.dof
        self.op += rhs.op
        self.N += rhs.N
        self.name += rhs.name
        return self

    def __add__(self, rhs):
        ret = self.copy()
        ret += rhs
        return ret

    def __sub__(self, rhs):
        ret = self.copy()
        ret += (-1) * rhs
        return ret

    def __neg__(self):
        return (-1) * self

    def __imul__(self, rhs):
        assert(self.dim == rhs.dim)
        self.x = [i + rhs.dof + 1 for i in self.x]
        self.y = [i + rhs.dof + 1 for i in self.y]
        self.x += rhs.x
        self.y += rhs.y
        self.dof += rhs.dof + 1
        self.op += rhs.op
        self.N += rhs.N
        self.name += rhs.name
        return self

    def __rmul__(self, lhs):
        ret = self.copy()
        ret.eval_op()
        for i in range(self.N):
            if ret.y[i] == 0:
                ret.op[i] *= lhs
        return ret

    def __div__(self, rhs):
        ret = self.copy()
        ret.eval_op()
        for i in range(self.N):
            if ret.y[i] == 0:
                ret.op[i] /= lhs
        return ret

    def __mul__(self, rhs):
        ret = self.copy()
        ret *= rhs
        return ret

    def __pow__(self, n):
        ret = self.copy()
        for i in range(n - 1):
            ret *= self
        return ret

    def copy(self):
        new = MPO()
        new.x = self.x.copy()
        new.y = self.y.copy()
        new.op = self.op.copy()
        new.name = self.name.copy()
        new.dof = self.dof
        new.N = self.N
        new.dim = self.dim
        return new
