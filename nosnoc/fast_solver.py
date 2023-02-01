from typing import Optional, List
from abc import ABC, abstractmethod
import time
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from nosnoc.problem import NosnocModel, NosnocOcp
from nosnoc.solver import NosnocSolverBase, get_results_from_primal_vector
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling
from nosnoc.utils import casadi_vertcat_list, casadi_sum_list, print_casadi_vector, casadi_length

def casadi_inf_norm_nan(x: ca.DM):
    norm = 0
    x = x.full().flatten()
    for i in range(len(x)):
        norm = max(norm, x[i])
    return norm


class NosnocFastSolver(NosnocSolverBase):
    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        super().__init__(opts, model, ocp)
        prob = self.problem

        # assume SCHOLTES_INEQ, or any mpcc_mode that transforms complementarities into inequalites.
        # loop over constraints and add duals
        #
        # problem of form
        # H(w) = 0
        # G_1(w) * G_2(w) = 0
        #                 \leq sigma (via Scholtes)
        # G_1(w), G_2(w) \geq 0

        # idea: apply FB on inequalities introduced in IP method on problem above.
        # optimality conditions:
        # - (\partial_w H)^\top \lambda_H
        # - \partial( G_1 * G_2 - \sigma - s) * lambda_comp
        # - \partial G_1 \top \mu_1 - \partial G_2 \top \mu_2
        # - \partial(s)^T * \mu_s = 0
        # \phi_{FB}(G_1, \mu_1, \tau) = 0
        # \phi_{FB}(G_2, \mu_2, \tau) = 0
        # \phi_{FB}(s, \mu_s, \tau) = 0

        # measure criteria: FB residuals, stationarity

        # complements
        G1 = casadi_vertcat_list([casadi_sum_list(x[0]) for x in prob.comp_list])
        G2 = casadi_vertcat_list([x[1] for x in prob.comp_list])
        # equalities
        H = casadi_vertcat_list( [prob.g[i] for i in range(len(prob.lbg)) if prob.lbg[i] == prob.ubg[i]] )

        n_comp = casadi_length(G1)
        n_H = casadi_length(H)

        # setup primal dual variables:
        lam_comp = ca.SX.sym('lam_comp', n_comp)
        lam_H = ca.SX.sym('lam_H', n_H)
        lam_pd = ca.vertcat(lam_comp, lam_H)
        self.lam_pd_0 = np.zeros((casadi_length(lam_pd),))

        mu_G1 = ca.SX.sym('mu_G1', n_comp)
        mu_G2 = ca.SX.sym('mu_G2', n_comp)
        mu_s = ca.SX.sym('mu_s', n_comp)
        mu_pd = ca.vertcat(mu_G1, mu_G2, mu_s)
        self.mu_pd_0 = np.ones((casadi_length(mu_pd),))

        slack = ca.SX.sym('slack', n_comp)
        w_pd = ca.vertcat(prob.w, slack, mu_pd, lam_pd)

        # slack = - G1 * G2 + sigma
        self.slack0_expr = -ca.diag(G1) @ G2 + prob.sigma
        self.slack0_fun = ca.Function('slack0_fun', [prob.w, prob.p], [self.slack0_expr])


        # barrier parameter
        tau = prob.tau
        sigma = prob.sigma

        # collect KKT system
        kkt_eq = H  # original equations
        slacked_complementarity = slack - self.slack0_expr
        kkt_eq = ca.vertcat(kkt_eq, slacked_complementarity)

        for i in range(n_comp):
            # treat IP complementarities:
            #  (G1, mu_1), (G2, mu_2), (s, mu_s) via FISCHER_BURMEISTER
            kkt_eq = ca.vertcat(kkt_eq, G1[i] + mu_G1[i] - ca.sqrt(G1[i]**2 + mu_G1[i]**2 + tau**2))
            kkt_eq = ca.vertcat(kkt_eq, G2[i] + mu_G2[i] - ca.sqrt(G2[i]**2 + mu_G2[i]**2 + tau**2))
            kkt_eq = ca.vertcat(kkt_eq, slack[i] + mu_s[i] - ca.sqrt(slack[i]**2 + mu_s[i]**2 + tau**2))

        # (\partial_w H)^\top \lambda_H
        # - \partial_w( G_1 * G_2 - \sigma - s) * lambda_comp
        # - \partial_w G_1 \top \mu_1 - \partial_w G_2 \top \mu_2
        # - \partial(s)^T * \mu_s = 0
        ws = ca.vertcat(prob.w, slack)

        stationarity = ca.jacobian(H, ws).T @ lam_H + \
             ca.jacobian(slacked_complementarity, ws).T @ lam_comp \
             - ca.jacobian(G1, ws).T @ mu_G1 \
             - ca.jacobian(G2, ws).T @ mu_G2 \
             - ca.jacobian(slack, ws).T @ mu_s
        kkt_eq = ca.vertcat(kkt_eq, stationarity)

        self.kkt_eq = kkt_eq
        self.kkt_eq_jac_fun = ca.Function('kkt_eq_jac_fun', [w_pd, prob.p], [kkt_eq, ca.jacobian(kkt_eq, w_pd)])
        self.kkt_eq_fun = ca.Function('kkt_eq_fun', [w_pd, prob.p], [kkt_eq])

        self.nw = casadi_length(prob.w)
        self.nw_pd = casadi_length(w_pd)
        print(f"created primal dual problem with {casadi_length(w_pd)} variables and {casadi_length(kkt_eq)} equations, {n_comp=}")


    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.

        :return: Returns a dictionary containing ... TODO document all fields
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()
        # slack0
        x0 = prob.model.x0
        p0 = prob.model.p_val_ctrl_stages[0]
        lambda00 = self.model.lambda00_fun(x0, p0).full().flatten()
        tau_val = opts.sigma_0
        p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([opts.sigma_0, tau_val]), lambda00, x0))
        slack0 = self.slack0_fun(prob.w0, p_val).full().flatten()
        w_pd_0 = np.concatenate((prob.w0, slack0, self.mu_pd_0, self.lam_pd_0))

        w_current = w_pd_0

        w_all = [w_current.copy()]
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
        p0 = prob.model.p_val_ctrl_stages[0]
        lambda00 = self.model.lambda00_fun(x0, p0).full().flatten()

        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:

        max_gn_iter = 12
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = sigma_k
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([sigma_k, tau_val]), lambda00, x0))

            t = time.time()
            for gn_iter in range(max_gn_iter):

                # get GN step
                kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, p_val)
                newton_matrix = jac_kkt_val + 1e-3 * ca.DM.eye(self.nw_pd)
                step = -ca.solve(jac_kkt_val, kkt_val)
                step_norm = casadi_inf_norm_nan(step)
                if step_norm < sigma_k:
                    break
                nlp_res = casadi_inf_norm_nan(kkt_val)

                # breakpoint()
                # print(f"{lambda00=}")
                # print("kkt_residual")
                # for ii in range(casadi_length(kkt_val)):
                #     print(f"{ii}\t {kkt_val[ii]} \t{self.kkt_eq[ii]}")

                # line search:
                alpha = 1.0
                rho = 0.5
                gamma = .3
                line_search_max_iter = 3
                for ls_iter in range(line_search_max_iter):
                    w_candidate = w_current + alpha * step.full().flatten()
                    step_res_norm = casadi_inf_norm_nan(self.kkt_eq_fun(w_current, p_val))
                    if step_res_norm < (1-gamma*alpha) * nlp_res:
                        break
                    else:
                        alpha *= rho

                cond = np.linalg.cond(newton_matrix.full())
                print(f"alpha = {alpha:.3f} \t step_norm {step_norm:.2f}\t cond(A) = {cond:.2e}")
                w_current = w_candidate

            cpu_time_nlp[ii] = time.time() - t

            # do line search
            # print and process solution
            status = 1
            nlp_iter[ii] = gn_iter # TODO
            cost_val = 0
            # nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            # cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_all.append(w_current)

            complementarity_residual = prob.comp_res(w_current[:self.nw], p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual
            if opts.print_level:
                self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
                                       cpu_time_nlp[ii], nlp_iter[ii], status)

            if complementarity_residual < opts.comp_tol:
                break

            if sigma_k <= opts.sigma_N:
                break

            # Update the homotopy parameter.
            if opts.homotopy_update_rule == HomotopyUpdateRule.LINEAR:
                sigma_k = opts.homotopy_update_slope * sigma_k
            elif opts.homotopy_update_rule == HomotopyUpdateRule.SUPERLINEAR:
                sigma_k = max(
                    opts.sigma_N,
                    min(opts.homotopy_update_slope * sigma_k,
                        sigma_k**opts.homotopy_update_exponent))

        # if opts.do_polishing_step:
        #     w_current, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish -
        #                                                      1] = self.polish_solution(
        #                                                          w_current, lambda00, x0)

        # collect results
        results = get_results_from_primal_vector(prob, w_current)

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_WOPT_PREV:
            prob.w0[:] = w_current[:]
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_current
        results["cost_val"] = 0.0

        # for i in range(len(w_current)):
        #     print(f"w{i}: {prob.w[i]} = {w_current[i]}")
        breakpoint()
        return results