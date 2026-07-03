import casadi as ca
from vdx_py.vector import *
from vdx_py.nlp import NLP
import nosnoc.mpccsol as mpccsol

class MPCC(NLP):
    def __init__(self,symbolic_type=ca.SX, name="nlp"):
        super().__init__(symbolic_type=symbolic_type, name=name)
        self.G = ConstraintVector(symbolic_type=symbolic_type)
        self.H = ConstraintVector(symbolic_type=symbolic_type)

    def to_casadi_dict(self):
        return {'x': self.w.sym,
                'g': self.g.sym,
                'p': self.p.sym,
                'G': self.H.sym,
                'H': self.p.sym,
                'f': self.f}

    def create_solver(self, mpccsol_opts, plugin="reg_homotopy"):
        # TODO(@anton) implement `mpccsol`
        self.solver = mpccsol.mpccsol(plugin, self, mpccsol_opts)

    def solve(self, casadi_opts=dict(), plugin="reg_homotopy"):
        if self.solver is None:
            self.create_solver(casadi_opts, plugin=plugin)

        mpcc_results = self.solver(x0=self.w.init,
                                 lbx=self.w.lb,
                                 ubx=self.w.ub,
                                 lbg=self.g.lb,
                                 ubg=self.g.ub,
                                 lam_g0=self.g.init_mult,
                                 lam_x0=self.w.init_mult,
                                 p=self.p.val)
        self.w.res = mpcc_results["w"]
        self.w.mult = mpcc_results["lam_x"]
        self.g.val = mpcc_results["g"]
        self.g.mult = mpcc_results["lam_g"]
        self.G.val = mpcc_results["G"]
        self.H.val = mpcc_results["H"]
        #
        self.f_result = mpcc_results['f']

        return self.solver.stats

    def __str__(self):
        ret = super.__str__(self)
        ret += self.G.__str__()
        ret += self.H.__str__()

        return ret
