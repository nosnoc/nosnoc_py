from .base import Base
from vdx_py.vector import *

import casadi as ca
import numpy as np

class Stewart(Base):

    def __init__(self, dcs, opts):
        super().__init__(dcs, opts)

    def _create_problem_parameters(self):
        pass

    @override
    def _create_variables(self):
        """Create Optimization Variables"""
        pass

    @override
    def _generate_direct_transcription_constraints(self):
        """Create direct transcription constraints"""
        pass

    @override
    def _generate_complementarity_constraints(self):
        """Create complementarity constraints"""
        pass

    @override
    def _generate_step_equilibration_constraints(self):
        """Create step equilibration constraints"""
        pass
