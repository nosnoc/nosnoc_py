import casadi as ca
import numpy as np
from itertools import product
from typing import List


class NosnocAutoModel:
    r"""
    An interface to automatically help generate a NosnocModel given:
    - x: symbolic state vector
    - f_nonsmooth_ode: symbolic vector field of the nonsmooth ODE
    Outputs the switching functions c as well as either (Not yet implemented):
    - F:
    - S:
    or:
    - alpha: Step reformulation multipliers used in the general inclusions expert mode.
    - f: x dot provided for general inclusions expert mode.

    Currently supported nonsmooth functions:
    - :math: `\mathrm{sign}(\cdot)`
    - :math: `\mathrm{max}(\cdot,\cdot)`
    - :math: `\mathrm{min}(\cdot,\cdot)`
    - TODO: more nonsmooth functions!
            In particular figure out a way to handle if_else
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

    def reformulate(self, force_nonlinear=True):
        if self._detect_nonlinearities() or force_nonlinear:
            self._reformulate_nonlin()
            return self.c, self.alpha, self.f
        else:
            self._reformulate_linear()
            return self.c, self.S, self.F

    def _reformulate_linear(self):
        # TODO actually implement linear reformulation
        raise Exception('Linear reformulation is not yet supported.')
        pass

    def _reformulate_nonlin(self):
        fs_nonsmooth = ca.vertsplit(self.f_nonsmooth_ode)
        fs = [self._rebuild_nonlin(f_nonsmooth) for f_nonsmooth in fs_nonsmooth]
        self.f = ca.vertcat(*fs)

    def _detect_nonlinearities(self):
        fs_nonsmooth = ca.vertsplit(self.f_nonsmooth_ode)
        for f_nonsmooth in fs_nonsmooth:
            nonlin_compontents = self._find_nonlinear_components(f_nonsmooth)
            for component in nonlin_compontents:
                nonlin, _ = self._find_nonlinearities(component)
                if nonlin:
                    return True
        return False

    # TODO type output
    def _rebuild_linear(self, node: ca.SX):
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

    def _rebuild_nonlin(self, node: ca.SX):
        if node.n_dep() == 0:
            # elementary variable
            return node
        elif node.n_dep() == 1:
            # unary operator
            if node.is_op(ca.OP_SIGN):
                # TODO need to process multiple layers, i.e. a nonsmooth function of nonsmooth inputs. Not handled for now
                #      but may be useful to detect at least.

                # get c and alpha for this nonsmoothness
                alpha, fresh = self._find_matching_expr_nonlin(node.dep(0))
                if fresh:
                    self.c.append(node.dep(0))
                    self.alpha.append(alpha)

                # calculate the equivalent expression for this node
                eq_expr = 2*alpha - 1
                return eq_expr
            else:
                # In this case do nothing
                child = self._rebuild_nonlin(node.dep(0))
                return ca.SX.unary(node.op(), child)
            # TODO: add more switching functions
        else:
            if node.is_op(ca.OP_FMAX):
                # get c and alpha for this nonsmoothness
                alpha, fresh = self._find_matching_expr_nonlin(node.dep(0)-node.dep(1))
                if fresh:
                    self.c.append(node.dep(0)-node.dep(1))
                    self.alpha.append(alpha)

                # calculate the equivalent expression for this node
                eq_expr = alpha*node.dep(0) + (1-alpha)*node.dep(1)
                return eq_expr
            elif node.is_op(ca.OP_FMIN):
                # get c and alpha for this nonsmoothness
                alpha, fresh = self._find_matching_expr_nonlin(node.dep(0)-node.dep(1))
                if fresh:
                    self.c.append(node.dep(0)-node.dep(1))
                    self.alpha.append(alpha)

                # calculate the equivalent expression for this node
                eq_expr = (1-alpha)*node.dep(0) + alpha*node.dep(1)
                return eq_expr
            else:
                # binary operator
                l_child = self._rebuild_nonlin(node.dep(0))
                r_child = self._rebuild_nonlin(node.dep(1))
                return ca.SX.binary(node.op(), l_child, r_child)

    def _find_matching_expr_nonlin(self, expr: ca.SX):
        found_c = None
        found_alpha = None
        # TODO: implement actual routine to find corresponding c
        if found_c is not None:
            return found_alpha, False
        else:
            # create a fresh alpha
            return ca.SX.sym(f'alpha_{len(self.alpha)}'), True

    def _find_nonlinearities(self, node: ca.SX):
        r'''
        Checks for nonsmooth nonlinearity, in order to auto select the reformulation used.
        :param node: Node in casADi graph to be checked for nonsmooth nonlinearity
        :returns:
        '''
        if node.n_dep() == 0:
            return False, False
        elif node.n_dep() == 1:
            if node.is_op(ca.OP_SIGN):
                nonlin, nonsmooth = self._find_nonlinearities(node.dep(0))
                if nonsmooth:
                    raise Exception('Nonsmooth functions of nonsmooth functions is not yet supported')
                return nonlin or nonsmooth, True
            else:
                nonlin, nonsmooth = self._find_nonlinearities(node.dep(0))
                return nonlin or nonsmooth, nonsmooth
        else:
            if node.is_op(ca.OP_FMAX):
                l_nonlin, l_nonsmooth = self._find_nonlinearities(node.dep(0))
                r_nonlin, r_nonsmooth = self._find_nonlinearities(node.dep(1))
                if l_nonsmooth or r_nonsmooth:
                    raise Exception('Nonsmooth functions of nonsmooth functions is not yet supported')
                return r_nonsmooth or l_nonsmooth, True
            elif node.is_op(ca.OP_FMIN):
                l_nonlin, l_nonsmooth = self._find_nonlinearities(node.dep(0))
                r_nonlin, r_nonsmooth = self._find_nonlinearities(node.dep(1))
                if l_nonsmooth or r_nonsmooth:
                    raise Exception('Nonsmooth functions of nonsmooth functions is not yet supported')
                return r_nonsmooth or l_nonsmooth, True
            else:
                l_nonlin, l_nonsmooth = self._find_nonlinearities(node.dep(0))
                r_nonlin, r_nonsmooth = self._find_nonlinearities(node.dep(1))
                return (l_nonsmooth and r_nonsmooth) or r_nonlin or l_nonlin, l_nonsmooth or r_nonsmooth

    def _find_nonlinear_components(self, node: ca.SX) -> List[ca.SX]:
        if node.n_dep() == 0:
            return [node]
        elif node.n_dep() == 1:
            if node.is_op(ca.OP_NEG):  # Note this is not used for rebuild so we dont care about polarity of components
                return self._find_nonlinear_components(node.dep(0))
            else:  # TODO: are there any other linear unary operators?
                return [node]
        else:
            if node.is_op(ca.OP_ADD):
                return self._find_nonlinear_components(node.dep(0)) + self._find_nonlinear_components(node.dep(1))
            elif node.is_op(ca.OP_SUB):
                return self._find_nonlinear_components(node.dep(0)) + self._find_nonlinear_components(node.dep(1))
            else:
                return [node]
