import casadi as ca
import numpy as np
from itertools import product


class NosnocAutoModel:
    """
    An interface to automatically generate a NosnocModel given:
    - x: symbolic state vector
    - f_nonsmooth_ode: symbolic vector field of the nonsmooth ODE
    Outputs the switching functions c as well as either:
    - F:
    - S:
    or:
    - alpha:
    - f:
    """

    def __init__(self,
                 x: ca.SX,
                 f_nonsmooth_ode: ca.SX):
        self.x = x
        self.f_nonsmooth_ode = f_nonsmooth_ode
        self.c = []
        self.S = []
        self.F = []
        self.f = []
        self.alpha = []

    def rebuild(self, node: ca.SX):
        if node.n_dep() == 0:
            # elementary variable
            return [node]
        elif node.n_dep() == 1:
            # unary operator
            if node.is_op(ca.OP_SIGN):
                # TODO need to process possibly multiple layers?
                #      but that would likely by a pathalogical case
                self.c.append(node.dep(0))
                self.S.append(np.array([[1], [-1]]))
                return [ca.SX(1), ca.SX(-1)]
            else:
                # In this case do nothing
                children = self.rebuild(node.dep(0))
                return [ca.SX.unary(node.op(), child) for child in children]
            # TODO: add more switching functions
        else:
            if node.is_op(ca.OP_FMAX):
                self.c.append(node.dep(0)-node.dep(1))
                self.S.append(np.array([[1], [-1]]))
                return [node.dep(0), node.dep(1)]
            elif node.is_op(ca.OP_FMIN):
                self.c.append(node.dep(0)-node.dep(1))
                self.S.append(np.array([[-1], [1]]))
                return [node.dep(0), node.dep(1)]
            else:
                # binary operator
                l_children = self.rebuild(node.dep(0))
                r_children = self.rebuild(node.dep(1))
                return [ca.SX.binary(node.op(), lc, rc) for lc, rc in product(l_children, r_children)]
