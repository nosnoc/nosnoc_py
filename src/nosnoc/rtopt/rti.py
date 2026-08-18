from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import override, List, Union
from time import monotonic

import numpy as np

from .rtopt import RealtimeOptimizationAlgorithm, RealtimeOptimizationStats, WarmstartType
from ..mpccsol.plugins.reg_homotopy import RegHomotopyOptions
from ..mpccsol.plugins.ccopt import CCOptOptions
from ..ocp import OcpSolver
from ..qpcc import Qpcc, ConvexificationOptions, ConvexificationMode

class PreparationStep(Enum):
    """
    An enum representing the several RTI Style algorithms.
    """

    NONE = auto()  #: HyRTI: No preparation step is taken other than re-linearizing.
    SQPCC = auto() #: AS-HyRTI: Some number of QPCCs are solved as predictors.
    FULL = auto()  #: Full-MPCC-HyRTI: The full nonlinar problem is solved during the preparation step.

@dataclass
class RTIOptions():
    """
    Options class for Real-Time Iteration style algorithms.
    """
    warmstart: WarmstartType = WarmstartType.SHIFT
    """
    How to warmstart the nonlinear problem, if we are solving it.
    """

    prepare_step: PreparationStep = PreparationStep.NONE
    """
    What kind of preparation step to take.
    """

    n_advanced_steps: int = 1
    """
    The number of SQP(CC) steps to take during the preparation phase.

    Info:
        This only does anything when `prepare_step == PreparationStep.SQPCC`.
    """

    cvx_opts: ConvexificationOptions = field(default_factory=ConvexificationOptions)
    """
    Convexification options for the QPCC solved in the `optimize` phase (and in the `prepare` phase if using an AS-RTI style algorithm).
    """

    gauss_newton_hessian: bool = True
    """
    Whether to use the Gauss-Newton Hessian (i.e. dropping constraint contributuion) or the exact Hessian.
    """

    use_complementarity_hessian: bool = True
    """
    Whether to use the MPCC Lagrangian or the to ignore the complementarity constraint multiplier contribution.
    """

    mpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)
    """
    Options for the solver which is used to solve the nonlinear problem.
    """

    qpcc_solver_opts: Union[RegHomotopyOptions,CCOptOptions] = field(default_factory=RegHomotopyOptions)
    """
    Options for the solver which is used to solve the QPCC.
    """

@dataclass
class RTIStats(RealtimeOptimizationStats):
    update_solve_time: List[float] = field(default_factory=list)
    """
    Time spent in the solver during the "feedback" (using the mpc language) phase.
    """

    advanced_step_solve_time: List[float] = field(default_factory=list)
    """
    Time spent in the solver during the "preparation" phase.
    """

class RTIAlgorithm(RealtimeOptimizationAlgorithm):

    def __init__(self, model, ocp_opts, rt_opts):
        super().__init__(model, ocp_opts, rt_opts)
        self.ocp_solver = OcpSolver(model, ocp_opts, rt_opts.mpcc_solver_opts)
        self.qpcc = Qpcc(
            self.ocp_solver.dtp,
            use_mpcc_multipliers=rt_opts.use_complementarity_hessian
        )
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
        np.copyto(self.ocp_solver.dtp.w.mult,self.qpcc.mpcc.w.mult)
        np.copyto(self.ocp_solver.dtp.g.mult,self.qpcc.mpcc.g.mult)
        np.copyto(self.ocp_solver.dtp.G.mult,self.qpcc.mpcc.G.mult)
        np.copyto(self.ocp_solver.dtp.G.mult,self.qpcc.mpcc.H.mult)

    @override
    def _update(self, **kwargs):
        if not self.initialized:
            self.last_converged = self._initial_solve(**kwargs)
            self.initialized = True
        else:
            self._measurement(**kwargs)
            self._take_qpcc_step()
            self.stats.update_solve_time.append(self.qpcc.solver.stats["wall_time_total"])
            self.last_converged = self.qpcc.solver.stats["converged"]
        return self._get_result()

    @override
    def _prepare(self, **kwargs):
        if self.rt_opts.prepare_step == PreparationStep.FULL:
           # Do initialization warmstart
            self.__warmstart_ocp()
            self._prediction(**kwargs)
            self.ocp_solver.solve()
            self.stats.advanced_step_solve_time.append(self.ocp_solver.dtp.solver.stats["wall_time_total"])
        elif self.rt_opts.prepare_step == PreparationStep.SQPCC:
            self._prediction(**kwargs)
            for ii in range(self.rt_opts.n_advanced_steps):
                # linearize at the current solution + predicted update
                advanced_step_solve_time = 0.0
                self.qpcc.linearize(
                    self.ocp_solver.dtp.w.res,
                    lam_g=self.ocp_solver.dtp.g.mult if not self.rt_opts.gauss_newton_hessian else None,
                    lam_G=self.ocp_solver.dtp.G.mult if not self.rt_opts.gauss_newton_hessian else None,
                    lam_H=self.ocp_solver.dtp.H.mult if not self.rt_opts.gauss_newton_hessian else None,
                    cvx_opts=self.rt_opts.cvx_opts,
                )
                self.qpcc.update_bounds()
                # solve a sinqle sqpcc step
                self._take_qpcc_step()
                advanced_step_solve_time += self.qpcc.solver.stats["wall_time_total"]
                # TODO(@anton): implement warmstart qpcc
            self.stats.advanced_step_solve_time.append(advanced_step_solve_time)
        else:
            self.stats.advanced_step_solve_time.append(0)
        self.qpcc.linearize(
            self.ocp_solver.dtp.w.res,
            lam_g=self.ocp_solver.dtp.g.mult if not self.rt_opts.gauss_newton_hessian else None,
            lam_G=self.ocp_solver.dtp.G.mult if not self.rt_opts.gauss_newton_hessian else None,
            lam_H=self.ocp_solver.dtp.G.mult if not self.rt_opts.gauss_newton_hessian else None,
            cvx_opts=self.rt_opts.cvx_opts
        )

    def __warmstart_ocp(self):
        if self.rt_opts.warmstart == WarmstartType.WARMSTART_PRIMALS:
            self.ocp_solver.warmstart(duals=False)
        elif self.rt_opts.warmstart == WarmstartType.WARMSTART_ALL:
            self.ocp_solver.warmstart(duals=True)
        elif self.rt_opts.warmstart == WarmstartType.SHIFT:
            self.ocp_solver.warmstart_shift()
