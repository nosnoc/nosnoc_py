import numpy as np

from .model import Pss
from .model import Cls
from .dcs import Stewart as StewartDCS
from .dcs import Heaviside as HeavisideDCS
from .dcs import Cls as ClsDCS
from .discrete_time_problem import Stewart as StewartDTP
from .discrete_time_problem import Heaviside as HeavisideDTP
from .discrete_time_problem import Cls as ClsDTP
from .nosnoc_types import DcsMode
from .mpccsol.plugins.reg_homotopy import RegHomotopyOptions
from .mpccsol.plugins.ccopt import CCOptOptions

class OcpSolver():

    def __init__(self, model, opts, solver_opts):
        self.model = model
        self.opts = opts
        self.solver_opts = solver_opts

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
        elif isinstance(model, Cls):
            self.dcs = ClsDCS(model, opts)
            self.dtp = ClsDTP(self.dcs, opts)
            self.dtp.populate_problem()
        else:
            raise NotImplementedError("Only Pss and Cls are implemented")

    def solve(self):
        self.set_param("rho_h",(), self.opts.rho_h)
        if isinstance(self.solver_opts, RegHomotopyOptions):
            plugin = "reg_homotopy"
        elif isinstance(self.solver_opts, CCOptOptions):
            plugin = "ccopt"
        else:
            raise NotImplementedError("Only reg_homotopy is implemented")

        return self.dtp.solve(casadi_opts=self.solver_opts, plugin=plugin)

    def _is_cls(self):
        return isinstance(self.model, Cls)

    def _has_lbp(self, var):
        """True if `var` is defined at the left boundary point, which only the state of a CLS is."""
        return self._is_cls() and (1,1,0) in var.ind_map

    def get(self, field): # TODO(@anton) allow for specialization in the DTP
        var = getattr(self.dtp.w, field) # TODO(@anton) try except
        if var.get_depth() == 3:
            if self._has_lbp(var):
                # Both boundary values of every finite element, see `_cls_trajectory`.
                return self._cls_trajectory(var)[1]
            end = self.opts.n_s+self.dtp.rbp
            return np.vstack([var[0,0,self.opts.n_s].res, var[1:,:,end].res])
        elif var.get_depth() == 2:
            return var[:,:].res
        elif var.get_depth() == 1:
            return var[:].res
        elif var.get_depth() == 0:
            return var[()].res

    def get_full(self, field):
        var = getattr(self.dtp.w, field) # TODO(@anton) try except
        if var.get_depth() == 3:
            return var[:,:,:].res
        elif var.get_depth() == 2:
            return var[:,:].res
        elif var.get_depth() == 1:
            return var[:].res
        elif var.get_depth() == 0:
            return var[()].res

    def set_param(self, field, index: tuple, value):
        param = getattr(self.dtp.p, field) # TODO(@anton) try except
        param[*index](val=value)

    def set_x0(self, x0):
        self.dtp.w.x[0,0,self.opts.n_s](lb=x0,ub=x0,init=x0)

    def _fe_lengths(self):
        """Lengths of all finite elements of the horizon, flattened over the control stages."""
        opts = self.opts
        if opts.use_fesd:
            return self.dtp.w.h[:,:].res

        h = self.dtp.p.T[()].val/(sum(self.opts.N_finite_elements))*(np.ones(sum(opts.N_finite_elements)))

        if self.opts.use_speed_of_time_variables:
            sot = self.get("sot")
            if self.opts.local_speed_of_time_variable:
                start = 0
                for ii,nfe in enumerate(self.opts.N_finite_elements):
                    h[start:start+nfe] = sot[ii]*h[start:start+nfe]
                    start += nfe
            else:
                h = sot*h
        return h

    def _cls_trajectory(self, var):
        """
        Trajectory of a Complementarity Lagrangian System including the post impact values.

        The velocity of a CLS is discontinuous at the finite element boundaries, so both boundary
        values of every finite element are reported, the post impact value `var[ii,jj,0]` and the
        pre impact value `var[ii,jj,n_s+rbp]`. They belong to the same physical time, therefore the
        finite element boundary times appear twice in the returned time grid, which makes the
        velocity jumps visible. The first finite element of a control stage only contributes its
        right boundary if impacts at the beginning of a stage are excluded, as there is no jump
        there.
        """
        opts = self.opts
        var_len = len(next(iter(var.ind_map.values())))
        n_fe = int(np.sum(opts.N_finite_elements))
        h = self._fe_lengths()

        var_lbp = np.reshape(var[1:,:,0].res, (n_fe, var_len))
        var_rbp = np.reshape(var[1:,:,opts.n_s+self.dtp.rbp].res, (n_fe, var_len))
        # Flattened indices of the first finite element of every control stage, the only elements
        # whose left boundary point carries no impulse when `no_initial_impacts` is set.
        stage_start = {int(ii) for ii in np.cumsum([0] + list(opts.N_finite_elements[:-1]))}

        var_out = [np.reshape(var[0,0,opts.n_s].res, (1, var_len))]
        t_out = [0.0]
        t = 0.0
        for jj in range(n_fe):
            if jj not in stage_start or not opts.no_initial_impacts:
                # post impact value at the left boundary of this finite element
                var_out.append(np.reshape(var_lbp[jj,:], (1, var_len)))
                t_out.append(t)
            var_out.append(np.reshape(var_rbp[jj,:], (1, var_len)))
            t = t + h[jj]
            t_out.append(t)
        return np.array(t_out), np.vstack(var_out)

    def get_time_grid(self):
        if self._is_cls():
            # The finite element boundary times appear twice, see `_cls_trajectory`.
            return self._cls_trajectory(self.dtp.w.x)[0]
        t_grid = np.cumsum(np.concatenate([[0], self._fe_lengths()]))
        return t_grid

    def get_time_grid_full(self):
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        t_grid_full = [np.array([0.0])]
        c = self.dtp.rk.colloc_points()
        h = self._fe_lengths()
        for jj in range(len(h)):
            start = t_grid_full[-1]
            if self._is_cls():
                # `get_full` reports the post impact value at the left boundary point, which shares
                # its time with the end of the previous finite element.
                t_grid_full.append(start)
            for kk in range(opts.n_s):
                t_grid_full.append(start + c[kk]*h[jj])
            if rbp:
                t_grid_full.append(start + h[jj])
        return np.concatenate(t_grid_full)

    def get_control_grid(self):
        if self.opts.use_fesd:
            h = self.dtp.w.h[:,:].res
        else:
            h = self.dtp.p.T[()].val/(np.sum(self.opts.N_finite_elements))*(np.ones(np.sum(self.opts.N_finite_elements)))

            if self.opts.use_speed_of_time_variables:
                sot = self.get("sot")
                h = sot*h
        t_grid = [0]
        for ii in range(1,self.opts.N_stages+1):
            h_sum = np.sum(self.dtp.w.h[ii,:].res) 
            sot = self.dtp._get_stage_sot(ii)
            h_sum *= sot
            t_grid.append(t_grid[-1]+h_sum)
        return np.array(t_grid)

    def get_objective(self):
        return self.dtp.f_result

    def get_w(self):
        return self.dtp.w.res

    def set(self, varname, indices, **kwargs):
        var = getattr(self.dtp.w, varname)
        var[*indices](**kwargs)

    def warmstart(self, duals=False):
        """
        Warmstart by copying the results vector into the init vector.
        If `duals==True` we do the same for all multipliers.
        """
        np.copyto(self.dtp.w.init, self.dtp.w.res)
        if duals:
            np.copyto(self.dtp.w.init_mult, self.dtp.w.mult)
            np.copyto(self.dtp.g.init_mult, self.dtp.g.mult)
            np.copyto(self.dtp.G.init_mult, self.dtp.G.mult)
            np.copyto(self.dtp.H.init_mult, self.dtp.H.mult)

    def warmstart_shift(self):
        """
        This method does a shift initialization by moving each control interval to the left by one.

        Warning:
            This is currently experimental and not guaranteed to work for all discretization settings.
        """
        self.dtp.warmstart_shift()
