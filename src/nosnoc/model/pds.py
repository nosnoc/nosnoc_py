from .base import Base, BaseDims
from ..dims import Dims

from typing import Optional, List
from numbers import Real

import casadi as ca
import numpy as np



class PdsDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_q = 0 
        self.n_c = 0


class Pds(Base):
    r"""Projected Dynamical System is a quasistatic model.
    This means we have no mass inertia. 
    Physically this means the velocity is negible in comparison to contact forces.
    
     """
    
  
    def __init__(self,
                 *,
                 f: ca.SX, #velocity of the system without constraints in x,u
                 f_c: ca.SX, #f_c is the gap function in x
                 J_n: Optional[ca.SX] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dims = PdsDims(self.dims)
        self.f = f
        self.f_c = f_c 
        self.friction_exists = False
        self.J_n = J_n
        
        self.__backfill()

    def __backfill(self):
        dims = self.dims

        dims.n_c = self.f_c.size(1)

        if self.J_n is None:
            self.J_n = ca.jacobian(self.f_c, self.x).T
    

    