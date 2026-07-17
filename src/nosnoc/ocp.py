import numpy as np

from .model import Pss
from .dcs import Stewart as StewartDCS
from .dcs import Heaviside as HeavisideDCS
from .discrete_time_problem import Stewart as StewartDTP
from .discrete_time_problem import Heaviside as HeavisideDTP
from .nosnoc_types import DcsMode
from .mpccsol.plugins.reg_homotopy import RegHomotopyOptions

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
        else:
            raise NotImplementedError("Only Pss is implemented")

    def solve(self):
        self.set_param("rho_h",(), self.opts.rho_h)
        if isinstance(self.solver_opts, RegHomotopyOptions):
            plugin = "reg_homotopy"
        else:
            raise NotImplementedError("Only reg_homotopy is implemented")

        return self.dtp.solve(casadi_opts=self.solver_opts, plugin=plugin)

    def get(self, field): # TODO(@anton) allow for specialization in the DTP
        var = getattr(self.dtp.w, field) # TODO(@anton) try except
        if var.get_depth() == 3:
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

    def get_time_grid(self):
        opts = self.opts
        if opts.use_fesd:
            h = self.dtp.w.h[:,:].res
        else:
            h = self.dtp.p.T[()].val/(sum(self.opts.N_finite_elements))*(np.ones(sum(opts.N_finite_elements)))

            if self.opts.use_speed_of_time_variables:
                sot = self.get("sot")
                h = sot*h
        t_grid = np.cumsum(np.concatenate([[0], h]))
        return t_grid

    def get_time_grid_full(self):
        opts = self.opts
        rbp = self.dtp.rbp
        dims = self.dcs.dims
        t_grid_full = [np.array([0.0])]
        c = self.dtp.rk.colloc_points()
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
        return np.concatenate(t_grid_full)

    def get_control_grid(self):
        if self.opts.use_fesd:
            h = self.dtp.w.h[:,:].res
        else:
            h = self.dtp.p.T[()].val/(sum(self.opts.N_finite_elements))*(np.ones(sum(opts.N_finite_elements)))

            if self.opts.use_speed_of_time_variables:
                sot = self.get("sot")
                h = sot*h
        t_grid = [0]
        for ii in range(1,self.opts.N_stages+1):
            h_sum = sum(self.dtp.w.h[ii,:].res)
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
