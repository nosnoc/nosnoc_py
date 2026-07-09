from .base import Base, BaseDims
from ..dims import Dims

from typing import Optional, List

import casadi as ca
import numpy as np

class HeavisideModelDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_alpha = 0


class Heaviside(Base):
    r"""
    A nonsmooth model which allows a more general than a PSS as it implements the Aizermann-Pyatnitskii extension.
    In particular it allows multipliers for example in the form :math:`1-\alpha_1\alpha_2`.

    TODO:
        Better document the details
    """
    def __init__(self,
                 f_x: ca.SX,
                 c: ca.SX,
                 alpha: ca.SX,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        assert alpha.size2() == 1, "alpha should be nx1"
        assert c.size2() == 1, "c should be nx1"
        assert alpha.size1() == c.size1(), "alpha and c should have the same length"
        assert f_x.size2() == 1, "f_x should be nx1"
        assert f_x.size1() == self.dims.n_x, "f_x should be same length as x"
        self.dims = HeavisideModelDims(self.dims)
        self.f_x = f_x
        self.c = c
        self.alpha = alpha
        self.dims.n_alpha = alpha.size1()
        self.__backfill()

    def __backfill(self):
        pass
