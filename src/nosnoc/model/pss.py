from .base import Base, BaseDims
from ..dims import Dims

from typing import Optional, List

import casadi as ca
import numpy as np

class PssDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_sys = 0
        self.n_c_sys = list()
        self.n_f_sys = list()


class Pss(Base):
    r"""
    A Piecewise smooth system writen via a (list of) matrix S and (list of) switching function c.
    Each region has dynamics defined by F.
    Alternatively one can define the regions via a Stewart type indicator function g_indicator.

    TODO:
        Better document the details
    """
    def __init__(self,
                 F: ca.SX|List[ca.SX],
                 S: Optional[np.ndarray|List[np.ndarray]] = None,
                 c: Optional[ca.SX|List[ca.SX]] = None,
                 g_indicator: Optional[ca.SX|List[ca.SX]] = None,
                 f_0: Optional[ca.SX] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dims = PssDims(self.dims)
        self.F = F
        self.S = S
        self.c = c
        self.g_indicator = g_indicator
        self.f_0 = f_0; self._populate_scalar("f_0", np.zeros(self.x.size(1)))
        self.__backfill()

    def __backfill(self):
        if isinstance(self.F, ca.SX): # Make self.F a list
            self.F = [self.F]
        if isinstance(self.S, np.ndarray): # Make self.S a list
            self.S = [self.S]
        n_sys = len(self.F)
        self.dims.n_sys = n_sys
        if self.g_indicator is None: # using S*c formulation
            self.g_indicator = list()
            if self.S is None: # S must exist
                raise RuntimeError("The switching matrix S, is not provided.")
            if len(self.S) != n_sys: # S must have n_sys elements
                raise RuntimeError("Number of matrices S does not match number of subsystems. Note that the number of subsystems is taken to be number of matrices F_i which collect the modes of every subsystem.")
            if self.c is None: # c must exist
                raise RuntimeError("Expresion for c, the constraint function for regions R_i is not provided.")
            if isinstance(self.c, ca.SX): # Make self.c a list
                self.c = [self.c]
            if len(self.c) != n_sys: # c must have n_sys elements
                raise RuntimeError("Number of different expressions for c does not match number of subsystems (taken to be number of matrices F_i which collect the modes of every subsystem).")
            for ii in range(n_sys):
                if self.S[ii].shape[1] != self.c[ii].size(1):
                    raise RuntimeError(f"The matrix S[{ii}] and vector c[{ii}] do not have compatible dimension.")
                self.g_indicator.append(-self.S[ii]@self.c[ii])
        elif isinstance(self.g_indicator, ca.SX):
            self.g_indicator = [self.g_indicator]

        if len(self.g_indicator) != n_sys:
            raise RuntimeError("Number of different expressions for g_indicator does not match number of subsystems (taken to be number of matrices F_i which collect the modes of every subsystem).")

        self.dims.n_c_sys = [c.size(1) for c in self.c]
        self.dims.n_f_sys = [f.size(2) for f in self.F]
