from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..model.base import Base as ModelBase

import casadi as ca
import numpy as np


class Base(ABC):
    r"""
    Base class for Dynamic Complementarity Systems of the (most generic) form.

    .. math::
        :nowrap:

        \begin{align*}
           \dot{x}&= f(x,z) \\
           0 &\le h(x,z) \perp z \ge 0
        \end{align*}
    """

    def __init__(self, model: ModelBase):
        self.model = model

    @abstractmethod
    def _generate_variables(self, opts):
        """Generate the required variables for the dcs"""
        pass

    @abstractmethod
    def _generate_expressions(self, opts):
        """Generate the required equations and functions for the dcs"""
        pass
