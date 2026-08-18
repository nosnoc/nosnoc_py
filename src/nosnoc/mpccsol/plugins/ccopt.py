from dataclasses import dataclass, field
from time import monotonic

import casadi as ca
import numpy as np
import nosnoc as ns
from vdx.vartypes import *
from .plugin import MpccsolPlugin
from .utils import find_nonscalar

@dataclass
class CCOptOptions():
    """
    Options for the `CCOpt` plugin for `mpccsol`.
    """
    madnlp_opts: dict = field(default_factory=lambda:
                              {
                                  "bound_relax_factor": 0.0
                              }
                              )
    """
    Dictionary containing `MadNLP` related options which are passed to `CCOpt`.
    """
    ccopt_opts: dict = field(default_factory=dict)
    """
    Dictionary containing `CCOpt` specific options.
    """



class CCOptSolver(MpccsolPlugin):
    """
    The `mpccsol` plugin which uses the MPCC tailored solver `CCOpt`.
    """
    @override
    def _build_solver(self):
        if isinstance(self.mpcc, dict):
            self.mpcc = ns.MPCC.from_casadi_dict(self.mpcc)
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
        start = monotonic()
        # TODO(@anton): Add support for multipliers, currently we don't support warmstarting the multipliers.
        res = self.solver(x0=x0,
                          lbx=lbx,
                          ubx=ubx,
                          lbg=np.concatenate((lbg, self.mpcc.G.lb[self.ind_nonscalar_G],self.mpcc.H.lb[self.ind_nonscalar_H])),
                          ubg=np.concatenate((ubg, self.mpcc.G.ub[self.ind_nonscalar_G],self.mpcc.H.ub[self.ind_nonscalar_H])),
                          p=p
                          )
        ng = len(self.mpcc.g)
        ncc = len(self.mpcc.G)
        G = np.zeros(ncc)
        G[self.ind_scalar_G] = np.reshape(res['x'].full()[self.ind_map_G], G[self.ind_scalar_G].shape)
        G[self.ind_nonscalar_G] = res['g'].full()[ng:ng+self.n_nonscalar_G].flatten()
        H = np.zeros(ncc)
        H[self.ind_scalar_H] = np.reshape(res['x'].full()[self.ind_map_H], H[self.ind_scalar_H].shape)
        H[self.ind_nonscalar_H] = res['g'].full()[ng+self.n_nonscalar_G:ng+self.n_nonscalar_G+self.n_nonscalar_H].flatten()
        mpcc_results = {
            "f": res['f'].full()[0],
            "w": res['x'].full().flatten(),
            "lam_x": res['lam_x'].full().flatten(),
            "g": res['g'].full()[:ng].flatten(),
            "lam_g": res['lam_g'].full()[:ng].flatten(), # TODO(@anton)use sorted indexing
            "G": G,
            "H": H,
        }
        self.stats["ccopt_stats"] = self.solver.stats()
        self.stats["wall_time_total"] = monotonic() - start
        self.stats["converged"] = self.solver.stats()["success"]
        return mpcc_results


    def _build_solver_impl(self):
        # Build vectors:
        ng = len(self.mpcc.g)
        ncc = len(self.mpcc.G)

        # Get nonscalar G
        ind_scalar_G, ind_nonscalar_G, ind_map_G = find_nonscalar(self.mpcc.G.sym, self.mpcc.w.sym, p=self.mpcc.p.sym)
        ind_scalar_H, ind_nonscalar_H, ind_map_H = find_nonscalar(self.mpcc.H.sym, self.mpcc.w.sym, p=self.mpcc.p.sym)

        n_nonscalar_G = len(ind_nonscalar_G)
        n_nonscalar_H = len(ind_nonscalar_H)

        # build ind_cc1
        ind_cc1 = np.zeros(ncc, int)
        ind_cc1[ind_scalar_G] = ind_map_G # Scalars should point to the variables themselves!
        ind_cc1[ind_nonscalar_G] = np.arange(ng, ng+n_nonscalar_G) # nonscalars are appened to g.
        # build ind_cc2
        ind_cc2 = np.zeros(ncc, int)
        ind_cc2[ind_scalar_H] = ind_map_H # Scalars should point to the variables themselves!
        ind_cc2[ind_nonscalar_H] = np.arange(ng+n_nonscalar_G, ng+n_nonscalar_G+n_nonscalar_H) # nonscalars ar appended to g.
        cc_pairs = np.vstack([ind_cc1, ind_cc2]).T
        # build cc_types
        cc_types = np.zeros(ncc,int)
        cc_types[ind_nonscalar_G] += 2
        cc_types[ind_nonscalar_H] += 1

        # save the index maps:
        self.ind_scalar_G = ind_scalar_G
        self.ind_nonscalar_G = ind_nonscalar_G
        self.ind_map_G = ind_map_G
        self.n_nonscalar_G = n_nonscalar_G

        self.ind_scalar_H = ind_scalar_H
        self.ind_nonscalar_H = ind_nonscalar_H
        self.ind_map_H = ind_map_H
        self.n_nonscalar_H = n_nonscalar_H

        self.ind_cc1 = ind_cc1
        self.ind_cc2 = ind_cc2
        self.cc_types = cc_types


        casadi_solver_opts = {
            "cc_pairs": cc_pairs.tolist(),
            "cc_types": cc_types.tolist(),
            "print_time": False,
        }
        if self.opts.ccopt_opts:
            casadi_solver_opts["ccopt"] = self.opts.ccopt_opts
        if self.opts.madnlp_opts:
            casadi_solver_opts["madnlp"] = self.opts.madnlp_opts
        nlp = {
            "x": self.mpcc.w.sym,
            "p": self.mpcc.p.sym,
            "f": self.mpcc.f,
            "g": ca.vertcat(self.mpcc.g.sym, self.mpcc.G.sym[ind_nonscalar_G,:], self.mpcc.H.sym[ind_nonscalar_H,:]),
        }
        self.solver = ca.nlpsol("solver", "ccopt", nlp, casadi_solver_opts)
