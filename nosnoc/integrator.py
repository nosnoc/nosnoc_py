from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, override
from warnings import warn

import numpy as np
import casadi as ca

from .model import Pss
from .dcs import Stewart as StewartDCS
from .discrete_time_problem import Stewart as StewartDTP
from .nosnoc_types import DcsMode
from nosnoc.mpccsol.plugins.reg_homotopy import RegHomotopyOptions

@dataclass
class IntegratorOptions():
    print_level: int = 0

@dataclass
class FESDIntegratorOptions(IntegratorOptions):
    solver_opts: RegHomotopyOptions = field(kw_only=True)
    use_previous_solution: bool = False

# TODO(@anton) implement smoothed integrator

class IntegratorPlugin(ABC):
    def __init__(self, model, opts, integrator_opts):
        self.model = model
        self.opts = opts
        self.integrator_opts = integrator_opts

    @abstractmethod
    def _solve(self):
        pass

    @abstractmethod
    def simulate(self, x0, u=None): #TODO(@anton) is kwargs the right abstraction here?
        pass

    @abstractmethod
    def get(self, field):
        pass

    @abstractmethod
    def get_full(self, field):
        pass

    @abstractmethod
    def get_time_grid(self):
        pass

    @abstractmethod
    def get_time_grid_full(self):
        pass

class FESDIntegratorPlugin(IntegratorPlugin):
    def __init__(self, model, opts, integrator_opts):
        super().__init__(model, opts, integrator_opts)
        self.w_all = []
        self.stats = []
        self.solver_opts = integrator_opts.solver_opts

        # TODO(@anton): add timefreezing here

        # do transform pipeline:
        if isinstance(model, Pss):
            if opts.dcs_mode == DcsMode.STEWART:
                self.dcs = StewartDCS(model)
                self.dtp = StewartDTP(self.dcs, opts)
                self.dtp.populate_problem()
            else:
                 raise NotImplementedError("Only Stewart is implemented")
        else:
            raise NotImplementedError("Only Pss is implemented")


    def _clear_history(self):
        self.w_all = []
        self.stats = []

    @override
    def _solve(self):
        self.set_param("rho_h",(), self.opts.rho_h)
        if isinstance(self.solver_opts, RegHomotopyOptions):
            plugin = "reg_homotopy"
        else:
            raise NotImplementedError("Only reg_homotopy is implemented")

        stats = self.dtp.solve(casadi_opts=self.solver_opts, plugin=plugin)
        self.stats.append(stats)
        self.w_all.append(self.dtp.w.res)
        return stats

    @override
    def simulate(self, x0, u=None):
        """
        Simultate the model for opts.N_sim step
        """
        # TODO(@anton) can x0 be optional
        # TODO(@anton) N_sim should live in integrator opts right?
        # TODO(@anton) asserts go away in production `-O` python calls, is this ok
        # TODO(@anton) do preallocation of np arrays
        assert u is None or (np.ndim(u)==2 and u.shape[0] == self.opts.N_sim and u.shape[1] == self.model.dims.n_u)
        assert np.ndim(x0)==1 and x0.shape[0] == self.model.dims.n_x

        opts = self.opts
        integrator_opts = self.integrator_opts

        x_res = [x0]
        x_res_full = [x0]
        t_grid = [0.0]
        t_grid_full = [0.0]

        # set x0
        self.dtp.w.x[0,0,self.opts.n_s](lb=x0,ub=x0,init=x0)
        self._clear_history()
        t_current = 0.0
        w0 = self.dtp.w.init
        rbp = self.dtp.rbp

        for ii in range(opts.N_sim):
            if u is not None: # Set control
                self.dtp.w.u[1](lb = u[ii,:], ub = u[ii,:], init = u[ii,:])

            solver_stats = self._solve()
            import pdb;pdb.set_trace()
            if not solver_stats["converged"]:
                constr_viol = solver_stats['constraint_violation']
                warn(f"integrator_fesd: did not converge in step {ii+1} constraint violation is: {constr_viol}")
            elif integrator_opts.print_level >= 2:
                wall_time_total = solver_stats["wall_time_total"]
                print(f"'Integration step {ii+1} / {opts.N_sim} ({t_current} s / {opts.N_sim*self.dtp.p.T[()].val} s) converged in {wall_time_total} s.")

            x_step = obj.discrete_time_problem.w.x(0,0,opts.n_s).res if rbp else np.array([])
            np.vstack(x_step, self.dtp.w.x[1:,:,opts.n_s+rbp].res)
            x_step_full = self.dtp.w.x[:,:,:].res
            x_res.append(x_step)
            x_res_full.append(x_step_full)
            if opts.use_fesd:
                h = self.dtp.w.h[:,:].res
            else:
                h = np.ones(opts.N_finite_elements[0]) * self.dtp.p.T[()].val/opts.N_finite_elements[0]
            t_grid.append(t_grid[-1] + np.cumsum(h))
            c = self.dtp.rk.collocation_points()
            for jj in range(len(h)):
                start = t_grid_full[-1]
                for kk in range(opts.n_s):
                    t_grid_full.append(start + c[kk]*h[jj])
                if rbp:
                    t_grid_full.append(start + h[jj])

             # warmstart solver
            if integrator_opts.use_previous_solution:
                np.copyto(self.dtp.w.init, self.dtp.w.res)

            self.dtp.w.x[0,0,self.opts.n_s](lb=x_step[-1,:],ub=x_step[-1,:],init=x_step[-1,:])

        return t_grid, x_res, t_grid_full, x_res_full



    @override
    def get(self, field):
        pass


    @override
    def get_full(self, field):
        pass

    @override
    def get_time_grid(self, field):
        pass

    @override
    def get_time_grid_full(self, field):
        pass

    def set_param(self, field, index: tuple, value):
        param = getattr(self.dtp.p, field) # TODO(@anton) try except
        param[*index](val=value)

class Integrator:
    def __init__(self, model, opts, integrator_opts):
        self.model = model
        self.opts = opts
        self.integrator_opts = integrator_opts

        if isinstance(integrator_opts, FESDIntegratorOptions):
            self.plugin = FESDIntegratorPlugin(model,opts,integrator_opts)
        else:
            raise NotImplementedError("Only FESD integrator is currently implemented")

    def simulate(self, x0, u=None): #TODO(@anton) is kwargs the right abstraction here?
        return self.plugin.simulate(x0, u=u)

    def get(self, field):
        return self.plugin.get(field)

    def get_full(self, field):
        return self.plugin.get_full(field)

    def get_time_grid(self):
        return self.plugin.get_time_grid()

    def get_time_grid_full(self):
        return self.plugin.get_time_grid_full()
