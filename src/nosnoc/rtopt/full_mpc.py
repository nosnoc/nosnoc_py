from dataclasses import dataclass, field
from enum import Enum, auto
from typing import override

import casadi as ca
import numpy as np

from .rtopt import RealtimeOptimizationAlgortihm, RealtimeOptimizationStats
from ..mpccsol.plugins.reg_homotopy import RegHomotopyOptions
from ..mpccsol.plugins.ccopt import CCOptOptions
from ..ocp import OcpSolver


class WarmstartType(Enum):
    NONE = auto()
    WARMSTART_PRIMALS = auto()
    WARMSTART_ALL = auto()
    SHIFT = auto()


@dataclass
class FullMPCOptions():
    warmstart: WarmstartType = WarmstartType.SHIFT
    mpcc_solver_opts = field(default_factory=RegHomotopyOptions)

@dataclass
class FullMPCStats(RealtimeOptimizationStats):
    optimize_solve_time: List[float] = field(default_factory=list)


class FullMPC(RealtimeOptimizationAlgorithm):
    def __init__(self, model, ocp_opts, rt_opts):
        super().__init__(model, ocp_opts, rt_opts)
        self.ocp_solver = OcpSolver(model, ocp_opts, rt_opts.mpcc_solver_opts)
        self.last_converged = False
        self.stats = FullMPCStats()

    @override
    def _optimize(self, x0):
        self.ocp_solver.set_x0(x0)
        self.ocp_solver.solve()
        self.stats.optimize_solve_time.push(self.ocp_solver.dtp.solver.stats["t_wall"])
        self.last_converged = self.ocp_solver.dtp.solver.stats["converged"]
        return self.ocp_solver.dtp.w.u[1]

    @override
    def _prepare(self, x_pred):
        if self.last_converged:
            if self.rt_opts.warmstart == WarmstartType.WARMSTART_PRIMALS:
                self.ocp_solver.warmstart(duals=False)
            elif self.rt_opts.warmstart == WarmstartType.WARMSTART_ALL:
                self.ocp_solver.warmstart(duals=True)
            elif self.rt_opts.warmstart == WarmstartType.SHIFT:
                self.ocp_solver.warmstart_shift()
        else:
            pass
