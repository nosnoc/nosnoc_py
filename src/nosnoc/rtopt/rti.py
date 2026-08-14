from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import override, List, Union
from time import monotonic

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
class RTIOptions():
    warmstart: WarmstartType = WarmstartType.SHIFT
    prepare_step: PreparationStep = PreparationStep.NONE
    n_advanced_steps: int = 1
    mpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)
    qpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)

@dataclass
class RTIStats(RealtimeOptimizationStats):
    optimize_solve_time: List[float] = field(default_factory=list)
    advanced_step_solve_time: List[float] = field(default_factory=list)

class RTIAlgorithm(RealtimeOptimizationAlgorithm):

    def __init__(self, model, ocp_opts, rt_opts):
        super().__init__(model, ocp_opts, rt_opts)
        self.ocp_solver = OcpSolver(model, ocp_opts, rt_opts.mpcc_solver_opts)
        self.qpcc = Qpcc(self.ocp_solver.dtp)
        self.qpcc.create_solver(rt_opts.qpcc_solver_opts)
        self.initialized = False
        self.last_converged = False
        self.stats = RTIStats()

    @abstractmethod
    def _initial_solve(self, **kwargs):
        """
        Implements the initial solve for the RTI algorithm. In most cases, this is solving the full nonlinear problem.
        """
        pass

    @abstractmethod
    def _measurement(self, **kwargs):
        """
        Implements the parameter update in the optimize step of the RTI algorithm.
        """
        pass

    @abstractmethod
    def _prediction(self, **kwargs):
        """
        Implements the predicted parameter update in the prepare step of the RTI algorithm.
        """
        pass

    @abstractmethod
    def _get_result(self):
        """
        Returns the result for the given RTI algorithm. In the case of MPC this is the first control.
        """
        pass

    def _take_qpcc_step(self):
        self.qpcc.solve()
        self.ocp_solver.dtp.w.res += self.qpcc.mpcc.w.res

    @override
    def _optimize(self, **kwargs):
        if not self.initialized:
            self.last_converged = self._initial_solve(**kwargs)
            self.initialized = True
        else:
            self._measurement(**kwargs)
            self.qpcc.solve()
            self.__update_with_qpcc_sol()
            self.stats.optimize_solve_time.append(self.qpcc.solver.stats["t_wall"])
            self.last_converged = self.qpcc.solver.stats["converged"]
        return self._get_result()

    @override
    def _prepare(self, **kwargs):
        if self.rt_opts.prepare_step == PreparationStep.FULL:
           # Do initialization warmstart
            self.__warmstart_ocp()
            self._prediction(**kwargs)
            self.ocp_solver.solve()
        elif self.rt_opts.prepare_step == PreparationStep.SQPCC:
            for ii in range(self.rt_opts.n_advanced_steps):
                # linearize at the current solution + predicted update
                self._prediction(**kwargs)
                self.qpcc.linearize(self.ocp_solver.dtp.w.res)
                self.qpcc.update_bounds()
                # solve a sinqle sqpcc step
                self._take_qpcc_step()
                # TODO(@anton): implement warmstart qpcc
                
        self.qpcc.linearize(self.ocp_solver.dtp.w.res)

    def __warmstart_ocp(self):
        if self.rt_opts.warmstart == WarmstartType.WARMSTART_PRIMALS:
            self.ocp_solver.warmstart(duals=False)
        elif self.rt_opts.warmstart == WarmstartType.WARMSTART_ALL:
            self.ocp_solver.warmstart(duals=True)
        elif self.rt_opts.warmstart == WarmstartType.SHIFT:
            self.ocp_solver.warmstart_shift()

    def __update_with_qpcc_sol(self):
        self.ocp_solver.dtp.w.res += self.qpcc.mpcc.w.res
