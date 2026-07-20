from .base import Base, BaseDims
from ..dims import Dims

from typing import Optional, List

import casadi as ca
import numpy as np

class ClsDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_sys = 0 #system dimensions equal to dimension of generalized velocities, coordinates
        self.n_c_sys = list() #number of contact pairs
        


class Cls(Base):
    r"""
    A Piecewise smooth system writen via a (list of) matrix S and (list of) switching function c.
    Each region has dynamics defined by F.
    Alternatively one can define the regions via a Stewart type indicator function g_indicator.

    TODO:
        Better document the details
    """
    def __init__(self,
                 q: ca.SX, #Generalized coordinates $q\in\mathbb{R}^{n_q}
                 v: ca.SX, #Generalized velocities $v\in\mathbb{R}^{n_q}
                 M: np.ndarray, #Generalized inertia matrix. Can be a function of the state $q$
                 f_v: ca.SX, # Generalized acceleration $\dot{v} = f_v(x)\in\mathbb{R}^{n_q}$
                 f_c: ca.SX | list[ca.SX], #Contact gap functions $f_c(q)\in\mathbb{R}^{n_c}$
                 e: float | list[float],
                 inv_M: Optional[np.ndarray], #user provided inverse of the inertia matrix
                 J_normal: Optional[ca.SX], #Normal contact Jacobian $J_n$. This can be calculated automatically from the contact gap functions.
                 # TODO: Translate the following comment to python version
                 J_tangent: Optional[ca.SX], #Tangent contact Jacobian $J_t$. This must be provided if there is friction and using the Conic :attr:`~nosnoc.Options.friction_model`.

                 #This must be provided if there is friction and using the Polyhedral :mat:attr:`~nosnoc.Options.friction_model`.
                 #For every row $D_i$, $-D_i$ must also be a row in $D$.
                 D_t : Optional[ca.SX],
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dims = ClsDims(self.dims)
        self.q = q
        self.v = v
        self.M = M
        self.f_v = f_v
        self.f_c = f_c
        self.e = e
        self.inv_M = inv_M
        self.J_normal = J_normal
        self.J_tangent = J_tangent
        self.D_t = D_t

       
        
        self.__backfill()

    def __backfill(self):
        
        n_sys = self.v.size(1)
        self.dims.n_sys = n_sys
        if isinstance(self.f_c, ca.SX): # Make self.f_c a list
            n_c_sys = 1
            self.f_c = [self.f_c]
        else:
            n_c_sys = len(self.f_c)
        
        self.dims.n_c_sys = n_c_sys

        if self.inv_M is None: 
            self.inv_M = np.linalg.inv(self.M)
        
        
        







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
