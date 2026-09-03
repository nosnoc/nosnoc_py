from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, override
from warnings import warn

import numpy as np
import casadi as ca

from .model import Pss
from .model import Heaviside
from .model import Cls
from .dcs import Stewart as StewartDCS
from .dcs import Heaviside as HeavisideDCS
from .dcs import Cls as ClsDCS
from .discrete_time_problem import Stewart as StewartDTP
from .discrete_time_problem import Heaviside as HeavisideDTP
from .discrete_time_problem import Cls as ClsDTP
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
    use_previous_solution: bool = True
    # Guess for the contact multiplier when a failed CLS step is retried assuming an impact occurs.
    # Any positive value works, it only has to push the solver away from the non impacting branch of
    # the complementarity conditions.
    impact_guess_init: float = 7.0

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
        elif isinstance(model, Heaviside):
            self.dcs = HeavisideDCS(model)
            self.dtp = HeavisideDTP(self.dcs, opts)
            self.dtp.populate_problem()
        elif isinstance(model, Cls):
            self.dcs = ClsDCS(model, opts)
            self.dtp = ClsDTP(self.dcs, opts)
            self.dtp.populate_problem()
        else:
            raise NotImplementedError("Only Pss, Heaviside and Cls are implemented")

    def _is_cls(self):
        return isinstance(self.model, Cls)


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

    def _cls_step_result(self, h, t_start):
        """
        Collect the state trajectory of one CLS integration step, starting at time `t_start`.

        For every finite element both boundary states are reported, the post impact state
        `x[ii,jj,0]` and the pre impact state `x[ii,jj,n_s+rbp]`. They belong to the same physical
        time, therefore the finite element boundary times appear twice in the time grid, which makes
        the velocity jumps visible. The first finite element only contributes its right boundary if
        impacts at the beginning of a step are excluded, as there is no jump there.
        """
        opts = self.opts
        rbp = self.dtp.rbp
        n_x = self.model.dims.n_x

        x_lbp = np.reshape(self.dtp.w.x[1:,:,0].res, (opts.N_finite_elements[0], n_x))
        x_rbp = np.reshape(self.dtp.w.x[1:,:,opts.n_s+rbp].res, (opts.N_finite_elements[0], n_x))

        x_step = []
        t_step = []
        t = t_start
        for jj in range(opts.N_finite_elements[0]):
            if jj > 0 or not opts.no_initial_impacts:
                # post impact state at the left boundary of this finite element
                x_step.append(x_lbp[jj,:])
                t_step.append(t)
            x_step.append(x_rbp[jj,:])
            t = t + h[jj]
            t_step.append(t)
        return np.vstack(x_step), np.array(t_step)

    def _reset_friction_guess(self):
        """
        Zero the friction variables of the first control stage for the impact retry.

        The retry guesses a large normal impulse, so the tangential quantities from the failed
        attempt are meaningless; starting them at zero is consistent with a guess of "impact, not
        yet sliding". The variables that exist depend on the friction model, so they are discovered
        from the dcs stacks rather than listed again here.
        """
        if not self.model.friction_exists:
            return
        opts = self.opts
        variant = self.dtp.variant
        stage_names = [n for n in variant.z_alg_blocks if n not in ("lambda_normal", "y_gap")]
        impulse_names = [n for n in variant.z_impulse_blocks
                         if n not in ("Lambda_normal", "Y_gap", "P_vn", "N_vn")]
        for jj in range(1, opts.N_finite_elements[0]+1):
            for kk in range(1, opts.n_s+1):
                for name in stage_names:
                    getattr(self.dtp.w, name)[1,jj,kk](init=0.0)
        if opts.use_fesd and not self.dtp._is_relaxed_oc():
            for jj in range(2 if opts.no_initial_impacts else 1, opts.N_finite_elements[0]+1):
                for name in impulse_names:
                    getattr(self.dtp.w, name)[1,jj](init=0.0)

    def _retry_cls_step(self):
        """
        Re-solve the current step with an initial guess that assumes an impact occurs.

        The failed attempt is dropped from the history so that a retried step contributes a single
        entry, and the initial guess is restored afterwards so that the perturbation does not leak
        into the following steps.
        """
        opts = self.opts
        impact_guess = self.integrator_opts.impact_guess_init
        print("integrator_fesd: initial guess did not converge, retrying with an impact guess.")
        w_init = np.copy(self.dtp.w.init)

        start_fe = 2 if opts.no_initial_impacts else 1
        if self.dtp._is_relaxed_oc():
            # Patel's relaxed formulation has no impulse variables; the impact is a large contact
            # force, so guess that instead, using the same positive value as for the impulse.
            for jj in range(1, opts.N_finite_elements[0]+1):
                for kk in range(1, opts.n_s+1):
                    self.dtp.w.lambda_normal[1,jj,kk](init=impact_guess)
                    self.dtp.w.y_gap[1,jj,kk](init=0.0)
        else:
            for jj in range(start_fe, opts.N_finite_elements[0]+1):
                self.dtp.w.Lambda_normal[1,jj](init=impact_guess)
                self.dtp.w.Y_gap[1,jj](init=0.0)
                self.dtp.w.P_vn[1,jj](init=0.0)
                self.dtp.w.N_vn[1,jj](init=0.0)
            for jj in range(1, opts.N_finite_elements[0]+1):
                for kk in range(1, opts.n_s+1):
                    self.dtp.w.lambda_normal[1,jj,kk](init=0.0)
                    self.dtp.w.y_gap[1,jj,kk](init=0.0)
        self._reset_friction_guess()

        # Drop the failed attempt, then re-solve.
        self.stats.pop()
        self.w_all.pop()
        solver_stats = self._solve()

        np.copyto(self.dtp.w.init, w_init)
        return solver_stats

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
                if self._is_cls() and opts.use_fesd:
                    solver_stats = self._retry_cls_step()
                    if integrator_opts.print_level >= 2:
                        if not solver_stats["converged"]:
                            print(f"integrator_fesd: retry did not converge in step {ii+1} constraint violation is: {solver_stats['constraint_violation']}")
                        else:
                            print(f"Integration step {ii+1} / {integrator_opts.N_sim} ({t_current} s / {integrator_opts.N_sim*self.dtp.p.T[()].val} s) converged in {solver_stats['wall_time_total']} s.")
            elif integrator_opts.print_level >= 2:
                wall_time_total = solver_stats["wall_time_total"]
                print(f"'Integration step {ii+1} / {integrator_opts.N_sim} ({t_current} s / {integrator_opts.N_sim*self.dtp.p.T[()].val} s) converged in {wall_time_total} s.")

            if opts.use_fesd:
                h = self.dtp.w.h[:,:].res
            else:
                h = np.ones(opts.N_finite_elements[0]) * self.dtp.p.T[()].val/opts.N_finite_elements[0]

            if self._is_cls():
                # The velocity of a CLS is discontinuous at the finite element boundaries, so both
                # the post impact state (third index 0) and the pre impact state (third index
                # n_s+rbp) are reported, and the impact times appear twice in the time grid.
                x_step, t_step = self._cls_step_result(h, t_grid[-1][-1])
                t_grid.append(t_step)
            else:
                x_step = np.reshape(self.dtp.w.x[0,0,opts.n_s].res, (1, self.model.dims.n_x)) if rbp else np.empty((0,self.model.dims.n_x))
                x_int = np.reshape(self.dtp.w.x[1:,:,opts.n_s+rbp].res, (opts.N_finite_elements[0], self.model.dims.n_x))
                x_step = np.vstack([x_step, x_int])
                t_grid.append(t_grid[-1][-1] + np.cumsum(h))
            # Skipping the third index 0 only has an effect for a CLS, as the other discretizations
            # do not have a left boundary point.
            x_step_full = np.reshape(self.dtp.w.x[1:,:,1:].res, (opts.N_finite_elements[0]*n_steps, self.model.dims.n_x))
            x_res.append(x_step)
            x_res_full.append(x_step_full)
            t_current = t_grid[-1][-1]
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
        # Some variables (e.g. the CLS impulse variables under `no_initial_impacts`) are not defined
        # on every finite element, so the number of rows is derived from the index map.
        n_fe = len([k for k in var.ind_map.keys() if len(k) >= 2 and k[0] != 0]) if var.get_depth() == 2 else opts.N_finite_elements[0]
        var_shape = (n_fe, var_len)
        # Not every stage variable is also defined at the initial point. The CLS contact forces for
        # example only exist from the first finite element onwards.
        has_initial = (0,0,opts.n_s) in var.ind_map
        var_0 = np.reshape(var[0,0,opts.n_s].res, (1,var_len)) if var.get_depth() == 3 and has_initial else None
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
        dims = self.dcs.dims
        w_curr = np.copy(self.dtp.w.res)
        np.copyto(self.dtp.w.res, self.w_all[0])
        var =  getattr(self.dtp.w, field)
        var_len = len(next(iter(var.ind_map.values()))) # Assumes all are same length, we don't enforce this however
        # Derive the number of rows from the index map, as not every variable is defined on every
        # finite element or at every stage point.
        n_entries = len([k for k in var.ind_map.keys() if k[0] != 0])
        var_shape = (n_entries, var_len)
        # Not every stage variable is also defined at the initial point.
        has_initial = (0,0,opts.n_s) in var.ind_map
        var_0 = np.reshape(var[0,0,opts.n_s].res, (1,var_len)) if var.get_depth() == 3 and has_initial else None
        var_out = [] if var_0 is None else [var_0]
        for w in self.w_all:
            np.copyto(self.dtp.w.res, w)
            if var.get_depth() == 3:
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
                if self._is_cls():
                    # `get_full` reports the post impact state at the left boundary point, which
                    # shares its time with the end of the previous finite element.
                    t_grid_full.append(start)
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
