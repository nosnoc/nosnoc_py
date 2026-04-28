from typing import Optional, List
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
