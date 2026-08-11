import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, override
from copy import copy

import casadi as ca
import numpy as np
import nosnoc as ns
from vdx import NLP
from vdx.vartypes import *
from .plugin import MpccsolPlugin
from .relaxations import *

class HomotopyUpdateRule(Enum):
    LINEAR = auto() # sigma_k = homotopy_update_slope*sigma_N
    SUPERLINEAR = auto() # 'superlinear' - sigma_k = max(sigma_N,min(homotopy_update_k*slope_sigma,sigma_k^homotopy_update_exponent))

class HomotopySteeringStrategy(Enum):
    DIRECT = auto()
    ELL_1 = auto()
    ELL_INF = auto()

# TODO(@anton) make this a dataclass?
@dataclass
class RegHomotopyOptions():
    solver_name: str = 'nosnoc_solver'
    solver: str      = 'ipopt'

    # MPCC and Homotopy Settings
    complementarity_tol: float               = 1e-8
    objective_scaling_direct: bool           = True
    sigma_0: float                           = 1
    sigma_N: float                           = 1e-15
    homotopy_update_rule: HomotopyUpdateRule = HomotopyUpdateRule.LINEAR
    assume_lower_bounds: bool                = True
    lift_complementarities: bool             = False # TODO(@anton) Not implemented

    homotopy_update_slope: float           = 0.1
    homotopy_update_exponent: float        = 1.5 # the exponent in the superlinear rule
    N_homotopy                             = 10 # Maximum number of nlp solves
    s_elastic_max: float                   = 1e1
    s_elastic_min: float                   = 0.0
    s_elastic_0: float                     = 1.0
    decreasing_s_elastic_upper_bound: bool = True

    # Verbose
    print_level: int = 3

    # nlp solver Settings
    opts_casadi_nlp: dict = field(default_factory=lambda: {
        "print_time": 0,
        "verbose": False,
        "ipopt": {
            "sb"                      : 'yes',
            "print_level"             : 0,
            "max_iter"                : 3000,
            "bound_relax_factor"      : 0,
            "tol"                     : 1e-8,
            "dual_inf_tol"            : 1e-8,
            "dual_inf_tol"            : 1e-8,
            "compl_inf_tol"           : 1e-8,
            "acceptable_tol"          : 1e-6,
            "mu_strategy"             : 'adaptive',
            "mu_oracle"               : 'quality-function',
            "warm_start_init_point"   : 'yes',
            "linear_solver"           : 'mumps',
            "mumps_pivtol"            : 1e-4,
            "mumps_permuting_scaling" : 3
        }
        #snopt: {}
        #worhp: {}
        #uno: {}
    })

    #
    relaxation_strategy: MpccRelaxation                  = ScholtesRelaxation(inequality=True)
    homotopy_steering_strategy: HomotopySteeringStrategy = HomotopySteeringStrategy.DIRECT

    timeout_cpu: float  = 0
    timeout_wall: float = 0

    store_all_homotopy_iters: bool  = True # store every NLP solution in the homotopy loop;


class RegHomotopySolver(MpccsolPlugin):
    @override
    def _build_solver(self):
        self.f_relax = 0.0
        if isinstance(self.mpcc, dict):
            self._convert_dict_to_mpcc()
        self._build_solver_impl()

    def _update_nlp_vectors(
            self,
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
        self.nlp.w.init[self.rev_nlp_w_indmap[self.ind_w_mpcc]] = x0
        self.nlp.w.init_mult[self.rev_nlp_w_indmap[self.ind_w_mpcc]] = lam_x0
        self.nlp.w.lb[self.rev_nlp_w_indmap[self.ind_w_mpcc]] = lbx
        self.nlp.w.ub[self.rev_nlp_w_indmap[self.ind_w_mpcc]] = ubx
        self.nlp.g.lb[self.rev_nlp_g_indmap[self.ind_g_mpcc]] = lbg
        self.nlp.g.ub[self.rev_nlp_g_indmap[self.ind_g_mpcc]] = ubg
        self.nlp.g.init_mult[self.nlp_g_indmap[self.ind_g_mpcc]] = lam_g0
        self.nlp.p.val[self.ind_p_mpcc] = p

    def _sigma_curr(self):
        return self.nlp.p.sigma[()].val[()]

    def _sigma(self):
        return self.nlp.p.sigma[()].sym

    def _comp_res_curr(self):
        return self.stats["comp_res"][-1]

    def _nlp_residual(self):
        return max(
            self.stats["nlp_stats"][-1]["iterations"]["inf_du"][-1],
            self.stats["nlp_stats"][-1]["iterations"]["inf_pr"][-1],
        )

    def _update_sigma(self):
        sigma_curr = self._sigma_curr()
        if self.opts.homotopy_update_rule == HomotopyUpdateRule.LINEAR:
            sigma_next = self.opts.homotopy_update_slope*sigma_curr
        elif self.opts.homotopy_update_rule == HomotopyUpdateRule.SUPERLINEAR:
            sigma_next = min(self.opts.homotopy_update_slope*sigma_curr, sigma_curr**self.opts.homotopy_update_exponent)
        self.nlp.p.sigma[()](val=sigma_next)

    def _prepare_nlp(self):
        np.copyto(self.nlp.w.init, self.nlp.w.res)
        self._update_sigma()

    def _solve_nlp(self):
        t_wall_start = time.time()
        stats = self.nlp.solve()
        t_wall_end = time.time()
        self.stats["t_wall"].append(t_wall_end - t_wall_start)
        self.stats["nlp_stats"].append(stats)
        comp_res = self.comp_res_fun(self.nlp.w.res, self.nlp.p.val).full()[0,0]
        self.stats["comp_res"].append(comp_res)
        if self.opts.print_level:
            self._print_iter_stats(
                self._sigma_curr(),
                comp_res,
                self._nlp_residual(),
                self.nlp.f_result,
                t_wall_end - t_wall_start,
                stats['iter_count'],
                stats['return_status']
            )

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
        self._update_nlp_vectors(x0,y0,lbx,ubx,lbg,ubg,p,lam_g0,lam_x0)

        self.stats = {
            "nlp_stats" : [],
            "t_wall" : [],
            "comp_res" : [self.comp_res_fun(self.nlp.w.init, self.nlp.p.val).full()[0,0]],
            "converged" : False
        }
        sigma_curr = self.opts.sigma_0
        self.nlp.p.sigma[()](val=sigma_curr)
        ii = 0
        if self.opts.print_level:
            self._print_header()
        while self._sigma_curr() >= self.opts.sigma_N and (ii<1 or self._comp_res_curr() > self.opts.complementarity_tol) and ii < self.opts.N_homotopy:
            self._solve_nlp()
            self._prepare_nlp()
            ii += 1

        G_val = self.G_mpcc_fun(self.nlp.w.res, self.nlp.p.val).full().flatten()
        H_val = self.H_mpcc_fun(self.nlp.w.res, self.nlp.p.val).full().flatten()

        comp_res = self.comp_res_fun(self.nlp.w.res, self.nlp.p.val).full()[0,0]
        last_stats = self.stats['nlp_stats'][-1]
        if last_stats['return_status'] in ("Solve_Succeeded", "Solved_to_Acceptable_Level") and comp_res <= self.opts.complementarity_tol:
            self.stats["converged"] = True
        else:
            self.stats["converged"] = False
        self.stats["wall_time_total"] = sum(self.stats["t_wall"])
        self.stats["constraint_violation"] = max(comp_res,self.stats["nlp_stats"][-1]["iterations"]["inf_pr"][-1])
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

    def _add_auxiliary_variables(self):
        self.nlp.p.sigma[()] = Parameter("sigma", 1)
        if self.opts.homotopy_steering_strategy == HomotopySteeringStrategy.ELL_INF:
            self.nlp.w.s_elastic[()] = Primal(
                "s_elastic", 1,
                lb=self.opts.s_elastic_min,
                ub=self.opts.s_elastic_max,
                init=self.opts.s_elastic_0,
            )
            if self.opts.decreasing_s_elastic_upper_bound:
                self.nlp.g.s_ub[()] = Constraint(self.nlp.w.s_elastic[()] - self.nlp.p.sigma[()], lb=-np.inf, ub=0.0)
            if self.opts.objective_scaling_direct:
                self.nlp.f += ca.inv(self.nlp.p.sigma[()].sym)*self.nlp.w.s_elastic[()]
            else:
                self.nlp.f = self.nlp.f*self.nlp.p.sigma[()] + self.nlp.w.s_elastic[()]

    def _get_relaxation_var(self, name, idx, length):
        if self.opts.homotopy_steering_strategy == HomotopySteeringStrategy.DIRECT:
            return np.ones(length)*self.nlp.p.sigma[()].sym
        elif self.opts.homotopy_steering_strategy == HomotopySteeringStrategy.ELL_INF:
            return np.ones(length)*self.nlp.w.s_elastic[()].sym
        elif self.opts.homotopy_steering_strategy == HomotopySteeringStrategy.ELL_1:
            getattr(self.nlp.w, f"s_{name}")[*idx] = Primal(
                f"s_{name}_{"_".join([str(i) for i in idx])}", length,
                lb=self.opts.s_elastic_min,
                ub=self.opts.s_elastic_max,
                init=self.opts.s_elastic_0,
            )
            s = getattr(self.nlp.w, f"s_{name}")[*idx].sym
            if self.opts.decreasing_s_elastic_upper_bound:
                getattr(self.nlp.g, f"s_{name}_ub")[*idx] = Constraint(s - self.nlp.p.sigma[()], lb=-np.inf, ub=0.0)
            if self.opts.objective_scaling_direct:
                self.nlp.f += ca.inv(self.nlp.p.sigma[()].sym)*ca.norm_1(s)
            else:
                self.f_relax += self.nlp.f*self.nlp.p.sigma[()] + ca.norm_1(s)
            return s

    def _convert_dict_to_mpcc(self):
        """
        If reg_homotopy is passed a dictionary, we convert it to an nosnoc MPCC object.
        This object inherits the lack of structure but allows us to reuse the vdx based handling in _build_solver.
        In order to minimize memory overhead, we use MX symbolics.
        """
        mpcc = ns.MPCC(symbolic_type=ca.MX)
        nx = self.mpcc["x"].size(1)
        np = self.mpcc["p"].size(1)
        f_fun = ca.Function("f", [self.mpcc["x"], self.mpcc["p"]], [self.mpcc["f"]])
        g_fun = ca.Function("g", [self.mpcc["x"], self.mpcc["p"]], [self.mpcc["g"]])
        G_fun = ca.Function("H", [self.mpcc["x"], self.mpcc["p"]], [self.mpcc["G"]])
        H_fun = ca.Function("G", [self.mpcc["x"], self.mpcc["p"]], [self.mpcc["H"]])

        mpcc.w.x[()] = Primal("x", nx)
        mpcc.p.p[()] = Parameter("p", np)
        mpcc.g.g[()] = Constraint(g_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.G.cc[()] = Constraint(G_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.H.cc[()] = Constraint(H_fun(mpcc.w.x[()], mpcc.p.p[()]))
        mpcc.f = f_fun(mpcc.w.x[()], mpcc.p.p[()])
        self.mpcc = mpcc

    def _build_solver_impl(self):
        """
        Build the regularization homotopy solver from a vdx MPCC class.
        """
        self.nlp = NLP(type(self.mpcc.f),name=f"relaxed_{self.mpcc.name}")
        self.nlp.f = self.mpcc.f
        self.nlp.w = copy(self.mpcc.w)
        self.nlp.g = copy(self.mpcc.g)
        self.nlp.p = copy(self.mpcc.p)
        self._add_auxiliary_variables()


        for (name,Gvar) in self.mpcc.G.variables.items():
            Hvar = self.mpcc.H.variables[name]
            for idx in Gvar.ind_map.keys():
                # TODO(@anton) do non scholtes
                length = Gvar[*idx].sym.size(1)
                relax, lb1, lb2 = self.opts.relaxation_strategy.relax(Gvar[*idx].sym, Hvar[*idx].sym, self._get_relaxation_var(name,idx,length))
                if not self.opts.assume_lower_bounds:
                    full_relax = ca.vertcat(relax[0], lb1[0], lb2[0])
                    full_lb = np.concatenate([relax[1], lb1[1], lb2[1]])
                    full_ub = np.concatenate([relax[2], lb1[2], lb2[2]])
                else:
                    full_relax = relax[0]
                    full_lb = relax[1]
                    full_ub = relax[2]
                getattr(self.nlp.g, f"{name}_relax")[*idx] = Constraint(full_relax, lb=full_lb, ub=full_ub)
        if not self.opts.objective_scaling_direct and self.opts.homotopy_steering_strategy == HomotopySteeringStrategy.ELL_1:
            self.nlp.f = self.nlp.f*self.nlp.p.sigma[()] + self.f_relax

        self.nlp_w_indmap,self.rev_nlp_w_indmap = self.nlp.w.resort_vector()
        self.nlp_g_indmap,self.rev_nlp_g_indmap = self.nlp.g.resort_vector()
        self.nlp_w_indmap,self.rev_nlp_w_indmap = np.array(self.rev_nlp_w_indmap),np.array(self.nlp_w_indmap)
        self.nlp_g_indmap,self.rev_nlp_g_indmap = np.array(self.rev_nlp_g_indmap),np.array(self.nlp_g_indmap)
        self.ind_w_mpcc = np.arange(0,len(self.mpcc.w))
        self.ind_g_mpcc = np.arange(0,len(self.mpcc.g))
        self.ind_p_mpcc = np.arange(0,len(self.mpcc.p))
        self.nlp.create_solver(self.opts.opts_casadi_nlp, plugin=self.opts.solver)

        # Build functions for data extraction:
        self.f_mpcc_fun = ca.Function("f_mpcc", [self.nlp.w.sym, self.nlp.p.sym], [self.mpcc.f])
        self.w_mpcc_fun = ca.Function("w_mpcc", [self.nlp.w.sym], [self.mpcc.w.sym])
        self.g_mpcc_fun = ca.Function("g_mpcc", [self.nlp.w.sym, self.nlp.p.sym], [self.mpcc.g.sym])
        self.G_mpcc_fun = ca.Function("G_mpcc", [self.nlp.w.sym, self.nlp.p.sym], [self.mpcc.G.sym])
        self.H_mpcc_fun = ca.Function("H_mpcc", [self.nlp.w.sym, self.nlp.p.sym], [self.mpcc.H.sym])
        self.comp_res_fun = ca.Function("comp_res", [self.nlp.w.sym, self.nlp.p.sym], [ca.mmax(self.mpcc.G.sym * self.mpcc.H.sym)])

    def _print_header(self):
        print('-------------------------------------------')
        print('sigma \t\t compl_res \t nlp_res \t cost_val \t CPU time \t iter \t status')

    def _print_iter_stats(self, sigma_k, complementarity_residual, nlp_res, cost_val, cpu_time_nlp, nlp_iter, status):
        print(f'{sigma_k:.1e} \t {complementarity_residual:.2e} \t {nlp_res:.2e}' +
              f'\t {cost_val:.2e} \t {cpu_time_nlp:3f} \t {nlp_iter} \t {status}')
