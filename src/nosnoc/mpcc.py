import casadi as ca
from vdx.vector import *
from vdx.nlp import NLP
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
                'G': self.G.sym,
                'H': self.H.sym,
                'f': self.f}

    def from_casadi_dict(mpcc_dict, symbolic_type=ca.SX):
        """
        Convert a raw CasADi dict form of an MPCC to an MPCC object
        This object inherits the lack of structure but allows us to reuse code in e.g. `mpccsol`.

        Todo:
            Anton should figure out if the SX symbolics are always a sane default.
        """
        mpcc = MPCC(symbolic_type=symbolic_type)
        nx = mpcc_dict["x"].size(1)
        np = mpcc_dict["p"].size(1)
        f_fun = ca.Function("f", [mpcc_dict["x"], mpcc_dict["p"]], [mpcc_dict["f"]])
        g_fun = ca.Function("g", [mpcc_dict["x"], mpcc_dict["p"]], [mpcc_dict["g"]])
        G_fun = ca.Function("G", [mpcc_dict["x"], mpcc_dict["p"]], [mpcc_dict["G"]])
        H_fun = ca.Function("H", [mpcc_dict["x"], mpcc_dict["p"]], [mpcc_dict["H"]])

        mpcc.w.x[()] = Primal("x", nx)
        mpcc.p.p[()] = Parameter("p", np)
        mpcc.g.g[()] = Constraint(g_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.G.cc[()] = CConstraint(G_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.H.cc[()] = CConstraint(H_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.f = f_fun(mpcc.w.x[()], mpcc.p.p[()])
        return mpcc

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
        ret = ("NLP with Objective:\n"
               f"{self.f}\n"
               f"Variables:\n"
               f"{str(self.w)}\n"
               f"Parameters:\n"
               f"{str(self.p)}\n"
               f"Constraints:\n"
               f"{str(self.g)}\n"
               f"Complementarity G:\n"
               f"{str(self.G)}\n"
               f"Complementarity H:\n"
               f"{str(self.H)}\n"
               )
        return ret
