from dataclasses import dataclass, field
from enum import Enum, auto
from typing import override, List, Union

import casadi as ca
import numpy as np

from .rtopt import RealtimeOptimizationAlgorithm, RealtimeOptimizationStats, WarmstartType
from ..mpccsol.plugins.reg_homotopy import RegHomotopyOptions
from ..mpccsol.plugins.ccopt import CCOptOptions
from ..ocp import OcpSolver
from ..qpcc import Qpcc

class PreparationStep(Enum):
    NONE = auto()
    SQPCC = auto()
    FULL = auto()

@dataclass
class RTIMPCOptions():
    warmstart: WarmstartType = WarmstartType.SHIFT
    prepare_step: PreparationStep = PreparationStep.NONE
    mpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)
    qpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)

@dataclass
class RTIMPCStats(RealtimeOptimizationStats):
    optimize_solve_time: List[float] = field(default_factory=list)
    advanced_step_solve_time: List[float] = field(default_factory=list)


class RTIMPC(RealtimeOptimizationAlgorithm):
    def __init__(self, model, ocp_opts, rt_opts):
        super().__init__(model, ocp_opts, rt_opts)
        self.ocp_solver = OcpSolver(model, ocp_opts, rt_opts.mpcc_solver_opts)
        self.qpcc = Qpcc(self.ocp_solver.dtp)
        self.qpcc.create_solver(rt_opts.qpcc_solver_opts)
        self.initialized = False
        self.last_converged = False
        self.stats = RTIMPCStats()

    @override
    def _optimize(self, x0):
        if not self.initialized:
            self.ocp_solver.set_x0(x0)
            self.ocp_solver.solve()
            self.ocp_solver.warmstart(duals=True)
            self.last_converged = self.ocp_solver.dtp.solver.stats["converged"]
            self.initialized = True
        else:
            self.ocp_solver.set_x0(x0)
            self._set_qpcc_x0(x0)
            self.qpcc.solve()
            self._update_with_qpcc_sol()
            self.stats.optimize_solve_time.append(self.qpcc.solver.stats["t_wall"])
            self.last_converged = self.qpcc.solver.stats["converged"]
        return self.ocp_solver.dtp.w.u[1].res

    @override
    def _prepare(self, x_pred):
        # Initialize on the first time we call prepare by _always_ solving the full mpcc
        # Update qpcc linearization point and linearize

        # Do initialization warmstart
        if self.rt_opts.warmstart == WarmstartType.WARMSTART_PRIMALS:
            self.ocp_solver.warmstart(duals=False)
        elif self.rt_opts.warmstart == WarmstartType.WARMSTART_ALL:
            self.ocp_solver.warmstart(duals=True)
        elif self.rt_opts.warmstart == WarmstartType.SHIFT:
            self.ocp_solver.warmstart_shift()

        if self.rt_opts.prepare_step == PreparationStep.FULL:
            self.ocp_solver.set_x0(x_pred)
            self.ocp_solver.solve()
            print(f"u_as = {self.ocp_solver.dtp.w.u[1].res}")
        self.qpcc.linearize(self.ocp_solver.dtp.w.res)

    def _set_qpcc_x0(self, x0):
        self.qpcc.mpcc.w.x[0,0,self.ocp_solver.opts.n_s](lb=x0,ub=x0)
        self.qpcc.update_bounds()

    def _update_with_qpcc_sol(self):
        self.ocp_solver.dtp.w.res += self.qpcc.mpcc.w.res

    def get_predicted_state(self):
        return self.ocp_solver.dtp.w.x[1,:,:].res[-1,:].flatten()
