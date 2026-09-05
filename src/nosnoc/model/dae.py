from .base import Base, BaseDims
from ..dims import Dimes

from typing import Optional, List


import casadi as ca
import numpy as np

class DaeDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)

class Dae(Base):
    r"""
    A smooth Differential algebraic system. This essentially uses the base class with no additions :)
    """

    def __init__(self,
                 f_x:ca.SX,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if f_x.size(1) != self.dims.n_x:
            raise RuntimeError("the dimension of x_dot= f(x,z,u) does not match the state dimension.")
        if self.G_path.size(1) != 0 or self.H_path.size(1) != 0:
            raise RuntimeError("Dcs models do not support path complementarities.")
        self.f_x = f_x
