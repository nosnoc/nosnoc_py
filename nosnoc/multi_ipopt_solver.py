from typing import Optional

import casadi as ca
import numpy as np
import time


from .solver import NosnocSolverBase, get_results_from_primal_vector
from .model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts

from nosnoc.nosnoc_types import InitializationStrategy, PssMode, ConstraintHandling, Status
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.utils import flatten_layer, flatten, check_ipopt_success


class NosnocMIpoptSolver(NosnocSolverBase):
    """
    An alternative to the NosnocSolver that uses multiple Ipopt solvers for different steps of the homotopy.

    Each IPOPT solver has a fixed tolerance, and a fixed initial and target value for the barrier parameter.
    """

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        """Constructor.
        """
        super().__init__(opts, model, ocp)

        sigma = opts.sigma_0
        self.solvers = []
        while True:

            # create NLP Solver
            try:
                casadi_nlp = {
                    'f': self.problem.cost,
                    'x': self.problem.w,
                    'g': self.problem.g,
                    'p': self.problem.p
                }
                opts.tol_ipopt = sigma * 1e-1

                # https://github.com/casadi/casadi/wiki/FAQ%3A-Warmstarting-with-IPOPT
                opts.opts_casadi_nlp['ipopt']['mu_init'] = sigma * 1e-1
                opts.opts_casadi_nlp['ipopt']['mu_target'] = sigma * 1e-1
                opts.opts_casadi_nlp['ipopt']['warm_start_init_point'] = 'yes'
                opts.opts_casadi_nlp['ipopt']['warm_start_bound_push'] = 1e-4 * sigma
                opts.opts_casadi_nlp['ipopt']['warm_start_mult_bound_push'] = 1e-4 * sigma
                opts.opts_casadi_nlp['ipopt']['bound_relax_factor'] = 1e-2 * sigma ** 2
                opts.opts_casadi_nlp['ipopt']['mu_strategy'] = 'monotone'

                # NOTE: this would only work after first call
                # opts.opts_casadi_nlp['ipopt']['warm_start_same_structure'] = 'yes'

                self.solvers.append(ca.nlpsol(model.name, 'ipopt', casadi_nlp, opts.opts_casadi_nlp))
            except Exception as err:
                self.print_problem()
                print(f"{opts=}")
                print("\nerror creating solver for problem above.")
                raise err

            if sigma <= opts.sigma_N:
                break
            sigma = self.homotopy_sigma_update(sigma)

    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()

        w0 = prob.w0.copy()

        w_all = [w0.copy()]
        n_iter_polish = opts.max_iter_homotopy + (1 if opts.do_polishing_step else 0)
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]

        if opts.print_level:
            print('-------------------------------------------')
            print('sigma \t\t compl_res \t nlp_res \t cost_val \t CPU time \t iter \t status')

        sigma_k = opts.sigma_0

        if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
            lbw = prob.lbw.copy()
            ubw = prob.ubw.copy()

            # lambda00 != 0.0 -> corresponding thetas on first fe are zero
            I_active_lam = np.where(self.lambda00 > 1e1*opts.comp_tol)[0].tolist()
            ind_theta_fe1 = flatten_layer(prob.ind_theta[0][0], 2)  # flatten sys
            w_zero_indices = []
            for i in range(opts.n_s):
                tmp = flatten(ind_theta_fe1[i])
                try:
                    w_zero_indices += [tmp[i] for i in I_active_lam]
                except:
                    breakpoint()

            # if all but one lambda are zero: this theta can be fixed to 1.0, all other thetas are 0.0
            w_one_indices = []
            # I_lam_zero = set(range(len(self.lambda00))).difference( I_active_lam )
            # n_lam = sum(prob.model.dims.n_f_sys)
            # if len(I_active_lam) == n_lam - 1:
            #     for i in range(opts.n_s):
            #         tmp = flatten(ind_theta_fe1[i])
            #         w_one_indices += [tmp[i] for i in I_lam_zero]
            if opts.print_level > 1:
                print(f"fixing {prob.w[w_one_indices]} = 1. and {prob.w[w_zero_indices]} = 0.")
                print(f"Since self.lambda00 = {self.lambda00}")
            w0[w_zero_indices] = 0.0
            lbw[w_zero_indices] = 0.0
            ubw[w_zero_indices] = 0.0
            w0[w_one_indices] = 1.0
            lbw[w_one_indices] = 1.0
            ubw[w_one_indices] = 1.0

        else:
            lbw = prob.lbw
            ubw = prob.ubw

        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = min(sigma_k ** 1.5, sigma_k)
            # tau_val = sigma_k**1.5*1e3
            self.setup_p_val(sigma_k, tau_val)

            # solve NLP
            if ii > 0:
                sol = self.solvers[ii](x0=w0,
                                lbg=prob.lbg,
                                ubg=prob.ubg,
                                lbx=lbw,
                                ubx=ubw,
                                p=self.p_val,
                                lam_g0 = sol['lam_g'],
                                lam_x0 = sol['lam_x'])
            else:
                sol = self.solvers[ii](x0=w0,
                    lbg=prob.lbg,
                    ubg=prob.ubg,
                    lbx=lbw,
                    ubx=ubw,
                    p=self.p_val,)

            # statistics
            solver_stats = self.solvers[ii].stats()
            cpu_time_nlp[ii] = solver_stats['t_proc_total']
            status = solver_stats['return_status']
            nlp_iter[ii] = solver_stats['iter_count']
            nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            cost_val = ca.norm_inf(sol['f']).full()[0][0]

            # process iterate
            w_opt = sol['x'].full().flatten()
            w0 = w_opt
            w_all.append(w_opt)

            complementarity_residual = prob.comp_res(w_opt, self.p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual

            if opts.print_level:
                self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
                                       cpu_time_nlp[ii], nlp_iter[ii], status)
            if not check_ipopt_success(status):
                print(f"Warning: IPOPT exited with status {status}")

            if complementarity_residual < opts.comp_tol:
                break

            if sigma_k <= opts.sigma_N:
                break

            # Update the homotopy parameter.
            sigma_k = self.homotopy_sigma_update(sigma_k)

        if opts.do_polishing_step:
            # TODO: create a new solver for the polishing step!
            w_opt, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish - 1], status = \
                                            self.polish_solution(self.solvers[-1], w_opt)

        # collect results
        results = get_results_from_primal_vector(prob, w_opt)

        # print constraint violation
        if opts.print_level > 1 and opts.constraint_handling == ConstraintHandling.LEAST_SQUARES:
            threshold = np.max([np.sqrt(cost_val) / 100, opts.comp_tol * 1e2, 1e-5])
            g_val = prob.g_fun(w_opt, self.p_val).full().flatten()
            if max(abs(g_val)) > threshold:
                print("\nconstraint violations:")
                for ii in range(len(g_val)):
                    if abs(g_val[ii]) > threshold:
                        print(f"|g_val[{ii}]| = {abs(g_val[ii]):.2e} expr: {prob.g_lsq[ii]}")
                print(f"h values: {w_opt[prob.ind_h]}")
                # print(f"theta values: {w_opt[prob.ind_theta]}")
                # print(f"lambda values: {w_opt[prob.ind_lam]}")
                # print_casadi_vector(prob.g_lsq)

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_WOPT_PREV:
            prob.w0[:] = w_opt[:]
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_opt
        results["cost_val"] = cost_val

        if check_ipopt_success(status):
            results["status"] = Status.SUCCESS
        else:
            results["status"] = Status.INFEASIBLE

        return results
