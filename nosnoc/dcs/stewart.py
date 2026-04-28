from typing import Optional, List, override
from ..model import Pss as PssModel
from .base import Base

import casadi as ca
import numpy as np

class Stewart(Base):
    r"""
    Stewart reformulation of a PSS to a DCS.
    """
    def __init__(self, model:PssModel):
        super().__init__(model)

    @override
    def _generate_variables(self, opts):
        """Generate the required variables for the dcs"""
        pass

    @override
    def _generate_expressions(self, opts):
        """Generate the required equations and functions for the dcs"""
        pass
