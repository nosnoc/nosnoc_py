import casadi as ca
import numpy as np
import nosnoc as ns
from vdx import NLP
from vdx.vartypes import *
from .plugin import MpccsolPlugin

@dataclass
class CCOptOptions():
    madnlp_opts: dict = field(default_factory=lambda:
                              {
                                  "bound_relax_factor": 0.0
                              }
                              )
    ccopt_opts: dict = field(default_factory=dict)



class CCOptSolver(MpccsolPlugin):
    @override
    def _build_solver(self):
        self._build_solver_impl()

    @override
    def _solve(self,
               x0:     np.ndarray,
               y0:     np.ndarray,
               lbx:    np.ndarray,
               ubx:    np.ndarray,
               lbg:    np.ndarray,
               ubg:    np.ndarray,
               p:      np.ndarray,
               lam_g0: np.ndarray,
               lam_x0: np.ndarray,
               ):

        mpcc_results = {
            "f": self.f_mpcc_fun(self.nlp.w.res, self.nlp.p.val).full().flatten(),
            "w": self.w_mpcc_fun(self.nlp.w.res).full().flatten(),
            "lam_x": self.w_mpcc_fun(self.nlp.w.mult).full().flatten(),
            "g": self.g_mpcc_fun(self.nlp.w.res, self.nlp.p.val).full().flatten(),
            "lam_g": self.nlp.g.mult[self.ind_g_mpcc], # TODO(@anton)use sorted indexing
            "G": G_val,
            "H": H_val,
        }
        return mpcc_results

    def _build_solver_impl(self):
        # Build vectors:
        # TODO(@anton): port `find_nonscalar` because currently we treat everything as a nonlinear CC.
        #               this is inefficient.
        ng = len(self.mpcc.g)
        ncc = len(self.mpcc.g)
        ind_cc1 = np.arange(ng, ng+ncc)
        ind_cc2 = np.arange(ng+ncc, ng+2*ncc)
        cc_pairs = np.vstack(np.array(ind_cc1), np.array(ind_cc2))
        cc_types = 3*np.ones(len(self.mpcc.G))
        casadi_solver_opts = {
            "madnlp": self.opts.madnlp_opts,
            "ccopt": self.opts.ccopt_opts,
            "cc_pairs": cc_pairs,
            "cc_types": cc_types,
        }
        nlp = {
            "x": self.mpcc.w.sym,
            "p": self.mpcc.p.sym,
            "f": self.mpcc.f,
            "g": ca.vertcat(self.mpcc.g.sym, self.mpcc.G.sym, self.mpcc.H.sym),
        }
        self.solver = ca.nlpsol("solver", "ccopt", nlp, casadi_solver_opts)
