from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import nosnoc as ns
from vdx import NLP
from vdx.vartypes import *
from .plugin import MpccsolPlugin
from .utils import find_nonscalar

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

        # Get nonscalar G
        ind_scalar_G, ind_nonscalar_G, ind_map_G = find_nonscalar(self.mpcc.G.sym, self.mpcc.w.sym, p=self.mpcc.p.sym)
        ind_scalar_H, ind_nonscalar_H, ind_map_H = find_nonscalar(self.mpcc.H.sym, self.mpcc.w.sym, p=self.mpcc.p.sym)

        n_nonscalar_G = len(ind_nonscalar_G)
        n_nonscalar_H = len(ind_nonscalar_H)

        # build ind_cc1
        ind_cc1 = np.zeros(ncc)
        ind_cc1[ind_scalar_G] = ind_map_G # Scalars should point to the variables themselves!
        ind_cc1[ind_nonscalar_G] = np.arange(ng, n_nonscalar_G) # nonscalars are appened to g.
        # build ind_cc2
        ind_cc2 = np.zeros(ncc)
        ind_cc2[ind_scalar_H] = ind_map_H # Scalars should point to the variables themselves!
        ind_cc2[ind_nonscalar_H] = np.arange(ng+n_nonscalar_G, ng+n_nonscalar_G+n_nonscalar_H) # nonscalars ar appended to g.
        cc_pairs = np.vstack(ind_cc1, ind_cc2)
        # build cc_types
        cc_types = np.zeros(ncc)
        cc_types[ind_nonscalar_G] += 2
        cc_types[ind_nonscalar_H] += 1


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
