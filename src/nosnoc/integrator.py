from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, override
from warnings import warn

import numpy as np
import casadi as ca

from .model import Pss
from .dcs import Stewart as StewartDCS
from .dcs import Heaviside as HeavisideDCS
from .discrete_time_problem import Stewart as StewartDTP
from .discrete_time_problem import Heaviside as HeavisideDTP
from .nosnoc_types import DcsMode, RKRepresentation
from nosnoc.mpccsol.plugins.reg_homotopy import RegHomotopyOptions

@dataclass
class IntegratorOptions():
    N_sim: int
    T_sim: Optional[float] = None
    h_sim: Optional[float] = None
    print_level: int = 0

    def __post_init__(self):
        # Handle T_sim and h_sim
        if self.T_sim is not None and self.h_sim is None:
            self.h_sim = self.T_sim/self.N_sim
        elif self.T_sim is None and self.h_sim is not None:
            self.T_sim = self.h_sim*self.N_sim
        else:
            raise Exception("Please provide exactly one of T_sim and h_sim.")

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
            elif opts.dcs_mode == DcsMode.STEP:
                self.dcs = HeavisideDCS(model)
                self.dtp = HeavisideDTP(self.dcs, opts)
                self.dtp.populate_problem()
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
        self.w_all.append(np.copy(self.dtp.w.res))
        return stats

    @override
    def simulate(self, x0, u=None):
        """
        Simulate the model for integrator_opts.N_sim step
        """
        # TODO(@anton) can x0 be optional
        # TODO(@anton) asserts go away in production `-O` python calls, is this ok
        # TODO(@anton) do preallocation of np arrays
        assert u is None or (np.ndim(u)==2 and u.shape[0] == self.integrator_opts.N_sim and u.shape[1] == self.model.dims.n_u)
        assert np.ndim(x0)==1 and x0.shape[0] == self.model.dims.n_x

        opts = self.opts
        integrator_opts = self.integrator_opts

        x_res = [np.reshape(x0,(1, self.model.dims.n_x))]
        x_res_full = [np.reshape(x0,(1, self.model.dims.n_x))]
        t_grid = [np.array([0.0])]
        t_grid_full = [np.array([0.0])]

        # set x0
        self.dtp.w.x[0,0,self.opts.n_s](lb=x0,ub=x0,init=x0)
        self._clear_history()
        t_current = 0.0
        w0 = self.dtp.w.init
        rbp = self.dtp.rbp
        n_steps = (1 if opts.rk_representation == RKRepresentation.DIFFERENTIAL else opts.n_s+rbp)

        for ii in range(integrator_opts.N_sim):
            if u is not None: # Set control
                self.dtp.w.u[1](lb = u[ii,:], ub = u[ii,:], init = u[ii,:])

            solver_stats = self._solve()
            if not solver_stats["converged"]:
                constr_viol = solver_stats['constraint_violation']
                warn(f"integrator_fesd: did not converge in step {ii+1} constraint violation is: {constr_viol}")
            elif integrator_opts.print_level >= 2:
                wall_time_total = solver_stats["wall_time_total"]
                print(f"'Integration step {ii+1} / {integrator_opts.N_sim} ({t_current} s / {integrator_opts.N_sim*self.dtp.p.T[()].val} s) converged in {wall_time_total} s.")

            x_step = np.reshape(self.dtp.w.x[0,0,opts.n_s].res, (1, self.model.dims.n_x)) if rbp else np.empty((0,self.model.dims.n_x))
            x_int = np.reshape(self.dtp.w.x[1:,:,opts.n_s+rbp].res, (opts.N_finite_elements[0], self.model.dims.n_x))
            x_step = np.vstack([x_step, x_int])
            x_step_full = np.reshape(self.dtp.w.x[1:,:,:].res, (opts.N_finite_elements[0]*(n_steps), self.model.dims.n_x))
            x_res.append(x_step)
            x_res_full.append(x_step_full)
            if opts.use_fesd:
                h = self.dtp.w.h[:,:].res
            else:
                h = np.ones(opts.N_finite_elements[0]) * self.dtp.p.T[()].val/opts.N_finite_elements[0]
            t_grid.append(t_grid[-1][-1] + np.cumsum(h))
            c = self.dtp.rk.colloc_points()
            for jj in range(len(h)):
                start = t_grid_full[-1]
                if opts.rk_representation != RKRepresentation.DIFFERENTIAL:
                    for kk in range(opts.n_s):
                        t_grid_full.append(start + c[kk]*h[jj])
                if rbp:
                    t_grid_full.append(start + h[jj])

             # warmstart solver
            if integrator_opts.use_previous_solution:
                np.copyto(self.dtp.w.init, self.dtp.w.res)

            self.dtp.w.x[0,0,self.opts.n_s](lb=x_step[-1,:],ub=x_step[-1,:],init=x_step[-1,:])

        return np.concatenate(t_grid), np.vstack(x_res), np.concatenate(t_grid_full), np.vstack(x_res_full)



    @override
    def get(self, field):
        if not self.w_all:
            return None # TODO(@anton) probably raise an error instead
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        w_curr = np.copy(self.dtp.w.res)
        np.copyto(self.dtp.w.res, self.w_all[0])
        var = getattr(self.dtp.w, field)
        var_len = len(next(iter(var.ind_map.values()))) # Assumes all are same length, we don't enforce this however
        var_shape = (opts.N_finite_elements[0], var_len)
        var_0 = np.reshape(var[0,0,opts.n_s].res, (1,var_len)) if var.get_depth() == 3 else None
        var_out = [] if var_0 is None else [var_0]
        for w in self.w_all:
            np.copyto(self.dtp.w.res, w)
            if var.get_depth() == 3:
                end = opts.n_s+rbp
                try:
                    var_out.append(np.reshape(var[1:,:,end].res, var_shape))
                except:
                    raise Exception(f"Cannot get {field} as this value is not evaluated at the element end points")
            elif var.get_depth() == 2:
                var_out.append(np.reshape(var[1:,:].res, var_shape))
            elif var.get_depth() == 1:
                var_out.append(np.reshape(var[1:].res, (1,var_len)))
            elif var.get_depth() == 0:
                return var_out.append(np.reshape(var[()].res, (1,var_len)))

        np.copyto(self.dtp.w.res, w_curr)

        return np.vstack(var_out)



    @override
    def get_full(self, field):
        if not self.w_all:
            return None # TODO(@anton) probably raise an error instead
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        w_curr = np.copy(self.dtp.w.res)
        np.copyto(self.dtp.w.res, self.w_all[0])
        var = getattr(self.dtp.w, field)
        var_len = len(next(iter(var.ind_map.values()))) # Assumes all are same length, we don't enforce this however
        var_shape = (opts.N_finite_elements[0]*opts.n_s, var_len) if var.get_depth() == 3 else (opts.N_finite_elements[0], var_len)
        var_0 = np.reshape(var[0,0,opts.n_s].res, (1,var_len)) if var.get_depth() == 3 else None
        var_out = [] if var_0 is None else [var_0]
        for w in self.w_all:
            np.copyto(self.dtp.w.res, w)
            if var.get_depth() == 3:
                end = opts.n_s + rbp
                var_out.append(np.reshape(var[1:,:,:].res, var_shape))
            elif var.get_depth() == 2:
                var_out.append(np.reshape(var[1:,:].res, var_shape))
            elif var.get_depth() == 1:
                var_out.append(np.reshape(var[1:].res, (1,var_len)))
            elif var.get_depth() == 0:
                return var_out.append(np.reshape(var[()].res, (1,var_len)))

        np.copyto(self.dtp.w.res, w_curr)

        return np.vstack(var_out)

    @override
    def get_time_grid(self):
        if not self.w_all:
            return None # TODO(@anton) probably raise an error instead
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        w_curr = np.copy(self.dtp.w.res)
        np.copyto(self.dtp.w.res, self.w_all[0])
        t_grid = [np.array([0.0])]
        for w in self.w_all:
            np.copyto(self.dtp.w.res, w)
            if opts.use_fesd:
                h = self.dtp.w.h[:,:].res
            else:
                h = np.ones(opts.N_finite_elements[0]) * self.dtp.p.T[()].val/opts.N_finite_elements[0]
            t_grid.append(t_grid[-1][-1] + np.cumsum(h))

        np.copyto(self.dtp.w.res, w_curr)

        return np.concatenate(t_grid)


    @override
    def get_time_grid_full(self):
        if not self.w_all:
            return None # TODO(@anton) probably raise an error instead
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        w_curr = np.copy(self.dtp.w.res)
        np.copyto(self.dtp.w.res, self.w_all[0])
        t_grid_full = [np.array([0.0])]
        c = self.dtp.rk.colloc_points()
        for w in self.w_all:
            np.copyto(self.dtp.w.res, w)
            if opts.use_fesd:
                h = self.dtp.w.h[:,:].res
            else:
                h = np.ones(opts.N_finite_elements[0]) * self.dtp.p.T[()].val/opts.N_finite_elements[0]
            for jj in range(len(h)):
                start = t_grid_full[-1]
                for kk in range(opts.n_s):
                    t_grid_full.append(start + c[kk]*h[jj])
                if rbp:
                    t_grid_full.append(start + h[jj])

        np.copyto(self.dtp.w.res, w_curr)

        return np.concatenate(t_grid_full)

    def set_param(self, field, index: tuple, value):
        param = getattr(self.dtp.p, field) # TODO(@anton) try except
        param[*index](val=value)

class Integrator:
    def __init__(self, model, opts, integrator_opts):
        self.model = model
        self.opts = opts
        self.integrator_opts = integrator_opts

        self._update_opts()

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

    def _update_opts(self):
        """ Update nosnoc options with integrator options time parameters """
        self.opts.T = self.integrator_opts.h_sim
        self.opts.h = None
        self.opts.h_k = None
        self.opts._make_T_h_consistent()
        if isinstance(self.integrator_opts.solver_opts, RegHomotopyOptions):
            self.integrator_opts.solver_opts.print_level = 0 if self.integrator_opts.print_level < 3 else 4
