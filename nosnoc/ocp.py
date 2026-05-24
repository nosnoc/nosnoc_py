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
