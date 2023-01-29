from typing import Optional, List
from abc import ABC, abstractmethod
import time
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from nosnoc.nosnoc import NosnocModel, NosnocOcp, NosnocSolverBase, get_results_from_primal_vector
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling


class NosnocFastSolver(NosnocSolverBase):
    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        super().__init__(opts, model, ocp)
        prob = self.problem
        breakpoint()
        # self.fun_g_jac_gw = ca.Function('jac_g', [prob.w, prob.p], [prob.g, ca.jacobian(prob.g, prob.w)])


    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.

        :return: Returns a dictionary containing ... TODO document all fields
        """
        self.initialize()
        opts = self.opts
        prob = self.problem
        w_current = prob.w0.copy()

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

            for gn_iter in range(max_gn_iter):

                # get GN step
                g_val, jac_g_val = self.fun_g_jac_gw(w_current, p_val)
                A_gn = jac_g_val.T @ jac_g_val
                b_gn = -jac_g_val.T @ g_val
                # step = A_gn \ b_gn
                step_gn = ca.solve(A_gn, b_gn)
                step_norm = ca.norm_inf(step_gn)
                w_current += step_gn.full().flatten()
                # print(f"{step_norm=}")
                if step_norm < sigma_k:
                    break

            # do line search
            # breakpoint()
            # print and process solution
            status = 1
            nlp_iter[ii] = 0 # TODO
            # nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            # cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_all.append(w_current)

            complementarity_residual = prob.comp_res(w_current, p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual

            # if opts.print_level:
            #     self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
            #                            cpu_time_nlp[ii], nlp_iter[ii], status)

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
        # results["cpu_time_nlp"] = cpu_time_nlp
        # results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_current
        results["cost_val"] = 0.0

        # for i in range(len(w_current)):
        #     print(f"w{i}: {prob.w[i]} = {w_current[i]}")
        return results