from enum import Enum, auto
from typing import Optional, List, override
from copy import copy

import casadi as ca
import numpy as np
from vdx_py import MPCC, NLP
from vdx_py.vartypes import *
from .plugin import MpccsolPlugin

class HomotopyUpdateRule(Enum):
    LINEAR = auto() # sigma_k = homotopy_update_slope*sigma_N
    SUPERLINEAR = auto() # 'superlinear' - sigma_k = max(sigma_N,min(homotopy_update_k*slope_sigma,sigma_k^homotopy_update_exponent))

class MpccRelaxation(Enum):
    SCHOLTES_INEQ = auto()
    FISCHER_BURMEISTER_INEQ = auto()

class HomotopySteeringStrategy(Enum):
    DIRECT = auto()
    ELL_1 = auto()
    ELL_INF = auto()


class RegHomotopyOptions():
    def __init__(self):
        self.solver_name: str = 'nosnoc_solver'
        self.solver: str      = 'ipopt'

        # MPCC and Homotopy Settings
        self.complementarity_tol: float               = 1e-9
        self.objective_scaling_direct: bool           = True
        self.sigma_0: float                           = 1
        self.sigma_N: float                           = 1e-9
        self.homotopy_update_rule: HomotopyUpdateRule = HomotopyUpdateRule.LINEAR
        self.assume_lower_bounds: bool                = False;
        self.lift_complementarities: bool             = False;

        self.homotopy_update_slope: float           = 0.1
        self.homotopy_update_exponent: float        = 1.5 # the exponent in the superlinear rule
        self.N_homotopy                             = 0 # 0 -> set automatically
        self.s_elastic_max: float                   = 1e1
        self.s_elastic_min: float                   = 0
        self.s_elastic_0: float                     = 1
        self.decreasing_s_elastic_upper_bound: bool = 0

        # Verbose
        self.print_level: int = 3

        # nlp solver Settings
        self.opts_casadi_nlp: dict = {
            "print_time": 0,
            "verbose": False,
            "ipopt": {
                "sb"                    : 'yes',
                "print_level"           : 5,
                "max_iter"              : 3000,
                "bound_relax_factor"    : 0,
                "tol"                   : 1e-8,
                "dual_inf_tol"          : 1e-8,
                "dual_inf_tol"          : 1e-8,
                "compl_inf_tol"         : 1e-8,
                "acceptable_tol"        : 1e-6,
                "mu_strategy"           : 'adaptive',
                "mu_oracle"             : 'quality-function',
                "warm_start_init_point" : 'yes',
                "linear_solver"         : 'mumps',
            }
            #snopt: {}
            #worhp: {}
            #uno: {}
        }

        #
        self.relaxation_strategy: MpccRelaxation                  = MpccRelaxation.SCHOLTES_INEQ
        self.homotopy_steering_strategy: HomotopySteeringStrategy = HomotopySteeringStrategy.DIRECT

        self.timeout_cpu: float  = 0
        self.timeout_wall: float = 0

        self.store_all_homotopy_iters: bool  = True # store every NLP solution in the homotopy loop;
        self.normalize_homotopy_update: bool = True


class RegHomotopySolver(MpccsolPlugin):
    @override
    def _build_solver(self):
        if isinstance(self.mpcc, MPCC):
            self._build_solver_vdx()
        elif isinstance(self.mpcc, dict):
            self._build_solver_dict()


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
        self.nlp.w.init[self.ind_w_mpcc] = x0
        self.nlp.w.init_mult[self.ind_w_mpcc] = lam_x0
        self.nlp.w.lb[self.ind_w_mpcc] = lbx
        self.nlp.w.ub[self.ind_w_mpcc] = ubx
        self.nlp.g.lb[self.ind_g_mpcc] = lbg
        self.nlp.g.ub[self.ind_g_mpcc] = ubg
        self.nlp.g.init_mult[self.ind_g_mpcc] = lam_g0
        self.nlp.p.val[self.ind_p_mpcc] = p

        sigma_curr = self.opts.sigma_0
        while sigma_curr >= self.opts.sigma_N:
            self.nlp.p.sigma[()](val=sigma_curr)
            stats = self.nlp.solve()
            np.copyto(self.nlp.w.init, self.nlp.w.res)
            sigma_curr = sigma_curr*self.opts.homotopy_update_slope

    def _build_solver_vdx(self):
        """
        Build the regularization homotopy solver from a vdx_py MPCC class.
        """
        #TODO (@anton) reordering
        self.nlp = NLP(type(self.mpcc.f),name=f"relaxed_{self.mpcc.name}")
        self.nlp.f = self.mpcc.f
        self.nlp.w = copy(self.mpcc.w)
        self.nlp.g = copy(self.mpcc.g)
        self.nlp.p = copy(self.mpcc.p)
        self.nlp.p.sigma[()] = Parameter("sigma", 1)
        sigma = self.nlp.p.sigma[()].sym

        for (name,Gvar) in self.mpcc.G.variables.items():
            Hvar = self.mpcc.H.variables[name]
            for idx in Gvar.ind_map.keys():
                # TODO(@anton) do non sholtes
                getattr(self.nlp.g, f"{name}_relax")[*idx] = Constraint(Gvar[*idx]*Hvar[*idx] - sigma, lb=-np.inf, ub=0)

        self.ind_w_mpcc = np.arange(0,len(self.mpcc.w))
        self.ind_g_mpcc = np.arange(0,len(self.mpcc.g))
        self.ind_p_mpcc = np.arange(0,len(self.mpcc.p))
        self.nlp.create_solver(self.opts.opts_casadi_nlp, plugin=self.opts.solver)

    def _build_solver_dict(self):
        """
        Build the regularization homotopy solver from a vdx_py MPCC class.
        """
        pass
