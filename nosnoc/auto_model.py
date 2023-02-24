import casadi as ca
import numpy as np
from itertools import product
from collections import defaultdict
from typing import List
from nosnoc.utils import casadi_vertcat_list, casadi_sum_list


class NosnocAutoModel:
    r"""
    An interface to automatically generate a NosnocModel given:
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
    - :math: `\mathrm{max}(\cdot, \cdot)`
    - :math: `\mathrm{min}(\cdot, \cdot)`
    - TODO: more nonsmooth functions!
            In particular figure out a way to handle if_else (ca.OP_IF_ELSE_ZERO)
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
        fs_nonsmooth = ca.vertsplit(self.f_nonsmooth_ode)

        nsm = []
        sm = []
        for f_nonsmooth in fs_nonsmooth:
            nonlin_compontents = self._find_nonlinear_components(f_nonsmooth)
            smooth_components = []
            nonsmooth_components = []
            for nonlin_compontent in nonlin_compontents:
                smooth = self._check_smooth(nonlin_compontent)
                if smooth:
                    smooth_components.append(nonlin_compontent)
                else:
                    nonsmooth_components.append(nonlin_compontent)
            nsm.append(nonsmooth_components)
            sm.append(smooth_components)
        # each smooth component is part of the common dynamics this is added to the first element
        f_common = casadi_vertcat_list([ca.cumsum(f) for f in sm])

        # Extract all the Additive components and categorize them by switching function
        collated_components = []
        for components in nsm:
            collated = defaultdict(list)
            for component in components:
                f, c, S = self._rebuild_additive_component(component)
                collated[c.str()].append((f, c, S))
            collated_components.append(collated)

        all_c = set().union(*collated_components)

        cS_set = set()
        for ii, collated in enumerate(collated_components):
            for c_str in all_c:
                comps = collated[c_str]
                f_ij = [ca.SX(0), ca.SX(0)]
                for f, c, S in comps:
                    cS_set.add((c, S))
                    f_i[0] += f[0]
                    f_i[1] += f[1]
                

    def _reformulate_nonlin(self):
        fs_nonsmooth = ca.vertsplit(self.f_nonsmooth_ode)
        fs = [self._rebuild_nonlin(f_nonsmooth) for f_nonsmooth in fs_nonsmooth]
        self.f = ca.vertcat(*fs)

    def _detect_nonlinearities(self):
        fs_nonsmooth = ca.vertsplit(self.f_nonsmooth_ode)
        for f_nonsmooth in fs_nonsmooth:
            nonlin_compontents = self._find_nonlinear_components(f_nonsmooth)
            for component in nonlin_compontents:
                additive = self._check_additive(component)
                if additive:
                    return True
        return False

    # TODO type output
    def _rebuild_additive_component(self, node: ca.SX):
        if node.n_dep() == 0:
            # elementary variable
            return [node], None, None
        elif node.n_dep() == 1:
            # unary operator
            if node.is_op(ca.OP_SIGN):
                return [ca.SX(1), ca.SX(-1)], node.dep(0), np.array([[1], [-1]])
            else:
                # In this case do nothing
                children, c, S = self.rebuild(node.dep(0))
                return [ca.SX.unary(node.op(), child) for child in children], c, S
        else:
            if node.is_op(ca.OP_FMAX):
                return [node.dep(0), node.dep(1)], node.dep(0)-node.dep(1), np.array([[1], [-1]])
            elif node.is_op(ca.OP_FMIN):
                self.c.append(node.dep(0)-node.dep(1))
                self.S.append(np.array([[-1], [1]]))
                return [node.dep(0), node.dep(1)], node.dep(0)-node.dep(1), np.array([[-1], [1]])
            else:
                # binary operator
                l_children, l_c, l_S = self.rebuild(node.dep(0))
                r_children, r_c, r_S = self.rebuild(node.dep(1))
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
        :param node: Node in CasADi graph to be checked for nonsmooth nonlinearity
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

    def _check_additive(self, node: ca.SX) -> bool:
        if node.n_dep() == 1:
            # If negated go down one level
            if node.is_op(ca.OP_NEG):
                return self._check_additive(node.dep(0))
            elif node.is_op(ca.OP_SIGN):
                return True
        else:
            # If min or max. Then we are additive.
            if node.is_op(ca.OP_FMIN):
                return True
            elif node.is_op(ca.OP_FMAX):
                return True
            # If multiply figure out which one is nonsmooth and check the other one for smoothness
            elif node.is_op(ca.OP_MUL):
                if node.dep(0).is_op(ca.OP_SIGN) or node.dep(0).is_op(ca.OP_FMAX) or node.dep(0).is_op(ca.OP_FMIN):
                    return self._check_smooth(node.dep(1))
                elif node.dep(1).is_op(ca.OP_SIGN) or node.dep(1).is_op(ca.OP_FMAX) or node.dep(1).is_op(ca.OP_FMIN):
                    return self._check_smooth(node.dep(1))
                else:
                    return False  # NOTE: This is not a sufficient condition to say this doesn't work but for now this is ok.
            else:
                return False  # NOTE: This is not a sufficient condition to say this doesn't work but for now this is ok.
        pass

    def _check_smooth(self, node: ca.SX) -> bool:
        if node.n_dep() == 0:
            return True
        elif node.n_dep() == 1:
            if node.is_op(ca.OP_SIGN):
                return False
            else:
                smooth = self._check_smooth(node.dep(0))
                return smooth
        else:
            if node.is_op(ca.OP_FMAX):
                return False
            elif node.is_op(ca.OP_FMIN):
                return False
            else:
                l_smooth = self._check_smooth(node.dep(0))
                r_smooth = self._check_smooth(node.dep(1))
                return l_smooth and r_smooth

    def _find_nonlinear_components(self, node: ca.SX) -> List[ca.SX]:
        if node.n_dep() == 0:
            return [node]
        elif node.n_dep() == 1:
            if node.is_op(ca.OP_NEG):  # Note this is not used for rebuild so we dont care about polarity of components
                return [node.unary(ca.OP_NEG, component) for component in self._find_nonlinear_components(node.dep(0))]
            else:  # TODO: are there any other linear unary operators?
                return [node]
        else:
            if node.is_op(ca.OP_ADD):
                return self._find_nonlinear_components(node.dep(0)) + self._find_nonlinear_components(node.dep(1))
            elif node.is_op(ca.OP_SUB):
                return self._find_nonlinear_components(node.dep(0)) + [node.unary(ca.OP_NEG, component) for component in self._find_nonlinear_components(node.dep(1))]
            else:
                return [node]
