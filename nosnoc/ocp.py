import numpy as np

from .model import Pss
from .dcs import Stewart as StewartDCS
from .discrete_time_problem import Stewart as StewartDTP
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
            else:
                 raise NotImplementedError("Only Stewart is implemented")
        else:
            raise NotImplementedError("Only Pss is implemented")

    def solve(self):
        if isinstance(self.solver_opts, RegHomotopyOptions):
            plugin = "reg_homotopy"
        else:
            raise NotImplementedError("Only reg_homotopy is implemented")

        return self.dtp.solve(casadi_opts=self.solver_opts, plugin=plugin)

    def get(self, field):
        var = getattr(self.dtp.w, field) # TODO(@anton) try except
        if var.get_depth() == 3:
            end = self.opts.n_s
            return var[:,:,end:end+2].res
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
        if self.opts.use_fesd:
            h = self.dtp.w.h[:,:].res
        else:
            h = self.dtp.p.T[()].val/(sum(self.opts.N_finite_elements))*(np.ones(sum(opts.N_finite_elements)))

        if opts.use_speed_of_time_variables:
            sot = self.get("sot")
            h = sot*h
        t_grid = np.cumsum(np.concat([[0], h]))

    
