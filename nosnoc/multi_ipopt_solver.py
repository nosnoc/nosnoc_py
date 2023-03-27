from typing import Optional

import casadi as ca
import numpy as np
import time


from .solver import NosnocSolverBase, get_results_from_primal_vector
from .model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts

from nosnoc.nosnoc_types import InitializationStrategy, PssMode, HomotopyUpdateRule, ConstraintHandling, Status
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.rk_utils import rk4_on_timegrid
from nosnoc.utils import flatten_layer, flatten, get_cont_algebraic_indices, flatten_outer_layers, check_ipopt_success


class NosnocMIpoptSolver(NosnocSolverBase):
    """
    The nonsmooth problem is formulated internally based on the given options,
    dynamic model, and (optionally) the ocp data.
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
                opts.opts_casadi_nlp['ipopt']['mu_init'] = sigma * 1e-1
                opts.opts_casadi_nlp['ipopt']['mu_target'] = sigma * 1e-1
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

        :return: Returns a dictionary containing ... TODO document all fields
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()

        w0 = prob.w0.copy()

        w_all = [w0.copy()]
        n_iter_polish = opts.max_iter_homotopy + 1
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]

        if opts.print_level:
            print('-------------------------------------------')
            print('sigma \t\t compl_res \t nlp_res \t cost_val \t CPU time \t iter \t status')

        sigma_k = opts.sigma_0

        # lambda00 initialization
        x0 = prob.model.x0

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
            t = time.time()
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
            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            solver_stats = self.solvers[ii].stats()
            status = solver_stats['return_status']
            nlp_iter[ii] = solver_stats['iter_count']
            nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            cost_val = ca.norm_inf(sol['f']).full()[0][0]
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
            w_opt, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish - 1], status = \
                                            self.polish_solution(w_opt, self.lambda00, x0)

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

        # for i in range(len(w_opt)):
        #     print(f"w{i}: {prob.w[i]} = {w_opt[i]}")
        return results

    def polish_solution(self, w_guess, lambda00, x0):
        opts = self.opts
        prob = self.problem

        eps_sigma = 1e1 * opts.comp_tol

        ind_set = flatten(prob.ind_lam + prob.ind_lambda_n + prob.ind_lambda_p + prob.ind_alpha +
                          prob.ind_theta + prob.ind_mu)
        ind_dont_set = flatten(prob.ind_h + prob.ind_u + prob.ind_x + prob.ind_v_global +
                               prob.ind_v + prob.ind_z)
        # sanity check
        ind_all = ind_set + ind_dont_set
        for iw in range(len(w_guess)):
            if iw not in ind_all:
                raise Exception(f"w[{iw}] = {prob.w[iw]} not handled proprerly")

        w_fix_zero = w_guess < eps_sigma
        w_fix_zero[ind_dont_set] = False
        ind_fix_zero = np.where(w_fix_zero)[0].tolist()

        w_fix_one = np.abs(w_guess - 1.0) < eps_sigma
        w_fix_one[ind_dont_set] = False
        ind_fix_one = np.where(w_fix_one)[0].tolist()

        lbw = prob.lbw.copy()
        ubw = prob.ubw.copy()
        lbw[ind_fix_zero] = 0.0
        ubw[ind_fix_zero] = 0.0
        lbw[ind_fix_one] = 1.0
        ubw[ind_fix_one] = 1.0

        # fix some variables
        if opts.print_level:
            print(
                f"polishing step: setting {len(ind_fix_zero)} variables to 0.0, {len(ind_fix_one)} to 1.0."
            )
        for i_ctrl in range(opts.N_stages):
            for i_fe in range(opts.Nfe_list[i_ctrl]):
                w_guess[prob.ind_theta[i_ctrl][i_fe][:]]

            sigma_k, tau_val = 0.0, 0.0
            self.setup_p_val(sigma_k, tau_val)

            # solve NLP
            t = time.time()
            # TODO: create a new solver for the polishing step!
            sol = self.solvers[-1](x0=w_guess, lbg=prob.lbg, ubg=prob.ubg, lbx=lbw, ubx=ubw, p=self.p_val)
            cpu_time_nlp = time.time() - t

            # print and process solution
            solver_stats = self.solvers[-1].stats()
            status = solver_stats['return_status']
            nlp_iter = solver_stats['iter_count']
            nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_opt = sol['x'].full().flatten()

            complementarity_residual = prob.comp_res(w_opt, self.p_val).full()[0][0]
            if opts.print_level:
                self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
                                       cpu_time_nlp, nlp_iter, status)
            if status not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Warning: IPOPT exited with status {status}")

        return w_opt, cpu_time_nlp, nlp_iter, status
