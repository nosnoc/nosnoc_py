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
                 f_c: ca.SX, 
                 J_normal: Optional[ca.SX] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dims = PdsDims(self.dims)
        self.f_c = f_c #f_c is a function in x‚
        self.friction_exists = False
        self.J_normal = J_normal
        
        self.__backfill()

    def __backfill(self):
        dims = PdsDims()

        dims.n_c = self.f_c.size(1)

        if self.J_normal == None:
            self.J_normal = ca.jacobian(self.f_c, self.x).T
    

    