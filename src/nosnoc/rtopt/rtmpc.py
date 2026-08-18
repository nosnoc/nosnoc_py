from dataclasses import dataclass, field
from enum import Enum, auto
from typing import override, List, Union

import casadi as ca
import numpy as np

from .rti import RTIAlgorithm, RTIStats, WarmstartType, PreparationStep, RTIOptions

@dataclass
class RTIMPCOptions(RTIOptions):
    pass

@dataclass
class RTIMPCStats(RTIStats):
    pass


class RTIMPC(RTIAlgorithm):

    @override
    def _initial_solve(self, x0):
        """
        Implements the initial full nonlinear solve.
        """
        self.ocp_solver.set_x0(x0)
        self.ocp_solver.solve()
        self.ocp_solver.warmstart(duals=True)
        return self.ocp_solver.dtp.solver.stats["converged"]

    @override
    def _measurement(self, x0):
        """
        Implements the parameter update in the optimize step of the RTI algorithm.
        """
        self.ocp_solver.set_x0(x0)
        self._set_qpcc_x0(x0)

    @override
    def _prediction(self, x_pred):
        """
        Implements the predicted parameter update in the prepare step of the RTI algorithm.
        """
        self.ocp_solver.set_x0(x_pred)

    @override
    def _get_result(self):
        """
        Returns the result for the given RTI algorithm. In the case of MPC this is the first control.
        """
        return self.ocp_solver.dtp.w.u[1].res

    def _set_qpcc_x0(self, x0):
        self.qpcc.mpcc.w.x[0,0,self.ocp_solver.opts.n_s](lb=x0,ub=x0)
        self.qpcc.update_bounds()

    def get_predicted_state(self):
        return self.ocp_solver.dtp.w.x[1,:,:].res[-1,:].flatten()
