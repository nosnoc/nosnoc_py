from typing import Optional, List
from abc import ABC, abstractmethod
import time
import scipy
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from nosnoc.problem import NosnocModel, NosnocOcp
from nosnoc.solver import NosnocSolverBase, get_results_from_primal_vector
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling
from nosnoc.utils import casadi_vertcat_list, casadi_sum_list, print_casadi_vector, casadi_length, casadi_inf_norm_nan, flatten
from nosnoc.plot_utils import plot_matrix_and_qr, spy_magnitude_plot, spy_magnitude_plot_with_sign

DEBUG = False


def get_fraction_to_boundary(tau, current, step):
    alpha = 1
    n = len(step)
    for i in range(n):
        if step[i] < 0.0:
            alpha = min(- tau * current[i] / step[i], alpha)
    return alpha

class NosnocCustomSolver(NosnocSolverBase):
    def _setup_G(self) -> ca.SX:
        prob = self.problem
        ind_pos = sum([flatten(x) for x in [prob.ind_lam, prob.ind_theta, prob.ind_lambda_n, prob.ind_lambda_p, prob.ind_alpha, prob.ind_h]], [])
        ind_pos.sort()
        return prob.w[ind_pos]

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        super().__init__(opts, model, ocp)
        prob = self.problem

        # assume SCHOLTES_INEQ, or any mpcc_mode that transforms complementarities into inequalites.
        # complements
        G1 = casadi_vertcat_list([casadi_sum_list(x[0]) for x in prob.comp_list])
        G2 = casadi_vertcat_list([x[1] for x in prob.comp_list])
        # equalities
        H = casadi_vertcat_list( [prob.g[i] for i in range(len(prob.lbg)) if prob.lbg[i] == prob.ubg[i]] )

        self.H = H
        self.G1 = G1
        self.G2 = G2
        n_comp = casadi_length(G1)
        n_H = casadi_length(H)

        # G = (G1, G2) \geq 0
        self.G = self._setup_G()
        nG = casadi_length(self.G)


        # setup primal dual variables:
        lam_H = ca.SX.sym('lam_H', n_H)
        lam_comp = ca.SX.sym('lam_comp', n_comp)
        lam_pd = ca.vertcat(lam_H, lam_comp)
        self.lam_pd_0 = np.zeros((casadi_length(lam_pd),))

        mu_G = ca.SX.sym('mu_G', nG)
        mu_s = ca.SX.sym('mu_s', n_comp)
        mu_pd = ca.vertcat(mu_G, mu_s)
        self.mu_pd_0 = np.ones((casadi_length(mu_pd),))
        self.n_mu = nG + n_comp

        slack = ca.SX.sym('slack', n_comp)
        w_pd = ca.vertcat(prob.w, slack, lam_pd, mu_pd)

        self.w_pd = w_pd
        self.nw = casadi_length(prob.w)
        self.nw_pd = casadi_length(w_pd)

        ## KKT system
        # slack = - G1 * G2 + sigma
        self.slack0_expr = -ca.diag(G1) @ G2 + prob.sigma

        # barrier parameter
        tau = prob.tau

        # collect KKT system
        slacked_complementarity = slack - self.slack0_expr
        stationarity_w = ca.jacobian(prob.cost, prob.w).T \
            + ca.jacobian(H, prob.w).T @ lam_H \
            + ca.jacobian(slacked_complementarity, prob.w).T @ lam_comp \
            - ca.jacobian(self.G, prob.w).T @ mu_G \
            - ca.jacobian(slack, prob.w).T @ mu_s

        stationarity_s = ca.jacobian(slacked_complementarity, slack).T @ lam_comp \
             - ca.jacobian(slack, slack).T @ mu_s

        kkt_eq_without_comp = ca.vertcat(stationarity_w, stationarity_s, H, slacked_complementarity)

        # treat IP complementarities:
        kkt_comp = []
        for i in range(nG):
            kkt_comp = ca.vertcat(kkt_comp, self.G[i] * mu_G[i] - tau)
        for i in range(n_comp):
            kkt_comp = ca.vertcat(kkt_comp, slack[i] * mu_s[i] - tau)

        kkt_eq = ca.vertcat(kkt_eq_without_comp, kkt_comp)

        self.kkt_eq = kkt_eq
        kkt_eq_jac = ca.jacobian(kkt_eq, w_pd)

        # regularize kkt_compl jacobian?
        JIT = True
        casadi_function_opts = {}
        if JIT:
            casadi_function_opts = {"compiler": "shell", "jit": True, "jit_options": {"compiler": "gcc", "flags": ["-O3"]}}

        self.kkt_eq_jac_fun = ca.Function('kkt_eq_jac_fun', [w_pd, prob.p], [kkt_eq, kkt_eq_jac], casadi_function_opts)
        self.kkt_eq_fun = ca.Function('kkt_eq_fun', [w_pd, prob.p], [kkt_eq], casadi_function_opts)
        self.G_fun = ca.Function('G_fun', [prob.w, prob.p], [self.G], casadi_function_opts)
        self.slack0_fun = ca.Function('slack0_fun', [prob.w, prob.p], [self.slack0_expr], casadi_function_opts)

        self.n_comp = n_comp
        self.n_H = n_H
        kkt_block_sizes = [self.nw, n_comp, n_H, n_comp, nG, n_comp]
        self.kkt_eq_offsets = [0]  + np.cumsum(kkt_block_sizes).tolist()

        print(f"created primal dual problem with {casadi_length(w_pd)} variables and {casadi_length(kkt_eq)} equations, {n_comp=}, {self.nw=}, {n_H=}")

        return

    def print_iterate_threshold(self, iterate, threshold=1.0):
        for ii in range(self.nw_pd):
            if np.abs(iterate[ii]) > threshold:
                print(f"{ii}\t{self.w_pd[ii]}\t{iterate[ii]:.2e}")

    def print_iterate(self, iterate):
        for ii in range(self.nw_pd):
            print(f"{ii}\t{self.w_pd[ii]}\t{iterate[ii]:.2e}")

    def compute_step(self, matrix, rhs):
        # naive
        step = np.linalg.solve(matrix, rhs)
        return step

# Schur complement:
# [ A, B,
# C, D]
# * [x_1, x_2] = [y_1, y_2]
# ->
# with S = (A - BD^{-1}C):
# x1 = S^{-1}(y_1 - B D^{-1}y_2)
# x2 = D^-1 (y2 - C x_1)
    def compute_step_schur_np(self, matrix, rhs):
        nwh = self.nw + self.n_H
        A = matrix[:nwh, :nwh]
        B = matrix[:nwh, nwh:]
        C = matrix[nwh:, :nwh]
        D = matrix[nwh:, nwh:]
        y1 = rhs[:nwh]
        y2 = rhs[nwh:]

        # solve_D = scipy.sparse.linalg.factorized(D)
        # # D_inv_C = solve_D(C)
        # D_inv_C = np.zeros(C.shape)
        # for i in range(C.shape[1]):
        #     D_inv_C[:, i] = solve_D(C[:, i])
        D_inv_C = np.linalg.solve(D, C)

        S = A - B @ D_inv_C
        x1_rhs = y1 - B @ np.linalg.solve(D, y2)
        x1 = np.linalg.solve(S, x1_rhs)

        x2 = np.linalg.solve(D, y2 - C@x1)
        step = np.concatenate((x1, x2))

        return step

    def compute_step_schur_scipy(self, matrix, rhs):
        nwh = self.nw + self.n_H
        A = matrix[:nwh, :nwh]
        B = matrix[:nwh, nwh:]
        C = matrix[nwh:, :nwh]
        D = scipy.sparse.csc_matrix(matrix[nwh:, nwh:])
        y1 = rhs[:nwh]
        y2 = rhs[nwh:]

        solve_D = scipy.sparse.linalg.factorized(D)
        D_inv_C = np.zeros(C.shape)
        for i in range(C.shape[1]):
            D_inv_C[:, i] = solve_D(C[:, i])

        S = A - B @ D_inv_C
        x1_rhs = y1 - B @ solve_D(y2)
        x1 = np.linalg.solve(S, x1_rhs)

        x2 = solve_D(y2 - C@x1)
        step = np.concatenate((x1, x2))

        return step

    def check_newton_matrix(self, matrix):
        upper_left = matrix[:self.nw, :self.nw]
        lower_right = matrix[-self.n_mu:, -self.n_mu:]
        print(f"conditioning: upper left {np.linalg.cond(upper_left):.3e}, lower right {np.linalg.cond(lower_right):.3e}")
        return

    def get_mu(self, w_current):
        return w_current[-self.n_mu:]

    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()
        # slack0
        x0 = prob.model.x0
        lambda00 = prob.model.get_lambda00(opts)
        tau_val = opts.sigma_0
        p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([opts.sigma_0, tau_val]), lambda00, x0))

        lamH0 = 1.0 * np.ones((self.n_H,))
        lampd0 = np.concatenate((lamH0, np.ones(self.n_comp,)))

        # slack0 = self.slack0_fun(prob.w0, p_val).full().flatten()
        slack0 = np.ones((self.n_comp,))
        w_pd_0 = np.concatenate((prob.w0, lampd0, slack0, self.mu_pd_0))

        w_current = w_pd_0

        w_all = [w_current.copy()]
        n_iter_polish = opts.max_iter_homotopy + 1
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]
        sigma_k = opts.sigma_0
        # timers
        t_la = 0.0
        t_ls = 0.0
        t_ca = 0.0

        n_mu = self.n_mu
        # TODO: initialize duals ala Waechter2006, Sec. 3.6
        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
        if opts.print_level == 1:
            print(f"sigma\t\t iter \t res \t min_steps \t min_mu \t max mu \t min G")

        w_candidate = w_current.copy()
        max_gn_iter = 30
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = sigma_k
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([sigma_k, tau_val]), lambda00, x0))

            if opts.print_level > 1:
                print("alpha \t alpha_mu \t step norm \t kkt res \t min mu \t min G")
            t = time.time()
            alpha_min_counter = 0

            for gn_iter in range(max_gn_iter):

                t0_ca = time.time()
                kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, p_val)
                newton_matrix = jac_kkt_val.sparse()
                t_ca += time.time() - t0_ca
                newton_matrix = jac_kkt_val.full()

                # cond = np.linalg.cond(newton_matrix)
                # self.plot_newton_matrix(newton_matrix, title=f'newton matrix, cond(A) = {cond:.2e}', fig_filename=f'newton_matrix_{ii}_{gn_iter}.pdf')

                # if DEBUG:
                #     self.check_newton_matrix(newton_matrix)

                # regularize Lagrange Hessian
                newton_matrix[-n_mu:, -n_mu:] += 1e-5 * np.eye(n_mu)
                newton_matrix[:self.nw, :self.nw] += 1e-5 * np.eye(self.nw)
                # self.plot_newton_matrix(newton_matrix, title=f'regularized matrix', )
                rhs = - kkt_val.full().flatten()


                # compute step
                # step = self.compute_step_schur_np(newton_matrix, rhs)
                # step = self.compute_step_schur_scipy(newton_matrix, rhs)
                t0_la = time.time()
                step = self.compute_step(newton_matrix, rhs)
                t_la += time.time() - t0_la

                step_norm = np.linalg.norm(step)
                nlp_res = casadi_inf_norm_nan(kkt_val)
                if step_norm < sigma_k or nlp_res < sigma_k / 10:
                    break

                ## LINE SEARCH
                t0_ls = time.time()
                # fraction to boundary
                tau_min = .99 # .99 is IPOPT default
                tau_j = max(tau_min, 1-tau_val)
                # tau_j = tau_min # dont use so large tau for now

                # compute alpha_mu_k
                mu_current = self.get_mu(w_current)
                mu_step = self.get_mu(step)
                alpha_mu = get_fraction_to_boundary(tau_j, mu_current, mu_step)
                w_candidate[-n_mu:] = mu_current + alpha_mu * mu_step

                # fraction to boundary G1, G2 > 0
                G_val = self.G_fun(w_current[:self.nw], p_val).full().flatten()
                G_delta_val = self.G_fun(step[:self.nw], p_val).full().flatten()
                alpha_k_max = get_fraction_to_boundary(tau_j, G_val, G_delta_val)

                # line search:
                alpha = alpha_k_max
                rho = 0.9
                gamma = .3
                alpha_min = 0.05
                while True:
                    w_candidate[:-n_mu] = w_current[:-n_mu] + alpha * step[:-n_mu]
                    step_res_norm = casadi_inf_norm_nan(self.kkt_eq_fun(w_candidate, p_val))
                    if step_res_norm < (1-gamma*alpha) * nlp_res:
                        break
                    elif alpha < alpha_min:
                        alpha_min_counter += 1
                        break
                    else:
                        alpha *= rho

                # if step_res_norm > nlp_res:
                #     print(f"alpha = {alpha:.2e}, alpha_k_max = {alpha_k_max:.2e}, alpha_mu = {alpha_mu:.2e}")
                #     print(f"residual increase from {nlp_res:.2e} to {step_res_norm:.2e}")
                    # breakpoint()

                # do step!
                w_current = w_candidate.copy()
                t_ls += time.time() - t0_ls


                # G_val_new = self.G_fun(w_current[:self.nw], p_val).full().flatten()
                # if any(G_val_new < 0):
                #     print("got G_val_new < 0")
                #     print(f"{alpha=}, {alpha_k_max=}")
                #     print(f"{G_val_new=}")
                #     breakpoint()

                if DEBUG:
                    maxA = np.max(np.abs(newton_matrix))
                    cond = np.linalg.cond(newton_matrix)
                    tmp = newton_matrix.copy()
                    tmp[tmp==0] = 1
                    minA = np.min(np.abs(tmp))
                    print(f"{alpha:.3f} \t {alpha_mu:.3f} \t {step_norm:.2e} \t {nlp_res:.2e} \t {cond:.2e} \t {maxA:.2e} \t {minA:.2e}")
                    # self.plot_newton_matrix(newton_matrix, title=f'regularized matrix: sigma = {sigma_k:.2e} cond = {cond:.2e}',
                    #         fig_filename=f'newton_spy_reg_{prob.model.name}_{ii}_{gn_iter}.pdf'
                    #  )
                elif opts.print_level > 1:
                    print(f"{alpha:.3f} \t {alpha_mu:.3f} \t {step_norm:.2e} \t {nlp_res:.2e} \t {min_mu:.2e}\t {np.min(G_val):.2e}\t")

            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            nlp_iter[ii] = gn_iter
            # nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            # cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_all.append(w_current)

            complementarity_residual = prob.comp_res(w_current[:self.nw], p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual
            if opts.print_level > 1:
                print(f"sigma = {sigma_k:.2e}, iter {gn_iter}, res {nlp_res:.2e}, min_steps {alpha_min_counter}")
            elif opts.print_level == 1:
                min_mu = np.min(self.get_mu(w_current))
                max_mu = np.max(self.get_mu(w_current))
                print(f"{sigma_k:.2e} \t {gn_iter} \t {nlp_res:.2e} \t {alpha_min_counter}\t {min_mu:.2e} \t {max_mu:.2e} \t {np.min(G_val):.2e}\t")

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

        sum_iter = sum([i for i in nlp_iter if i is not None])
        total_time = sum([i for i in cpu_time_nlp if i is not None])
        print(f"total iterations {sum_iter}, CPU time {total_time:.3f}: LA: {t_la:.3f} line search: {t_ls:.3f} casadi: {t_ca:.3f}")

        # self.print_iterate(w_current)
        return results

    def plot_newton_matrix(self, matrix, title='', fig_filename=None):
        import matplotlib.pyplot as plt
        from nosnoc.plot_utils import latexify_plot
        latexify_plot()
        fig, axs = plt.subplots(1, 1)
        self.add_scatter_spy_magnitude_plot(axs, fig, matrix)
        axs.set_title(title)

        # Q, R = np.linalg.qr(matrix)
        # axs[1].spy(Q)
        # axs[1].set_title('Q')
        # axs[2].spy(R)
        # axs[2].set_title('R')

        if fig_filename is not None:
            plt.savefig(fname=fig_filename)
            print(f"saved figure as {fig_filename}")
        plt.show()

    def add_scatter_spy_magnitude_plot(self, ax, fig, matrix):
        spy_magnitude_plot(matrix, ax=ax, fig=fig,
        # spy_magnitude_plot_with_sign(matrix, ax=ax, fig=fig,
            xticklabels=[r'$w$', r'$s$', r'$\lambda_H$', r'$\lambda_{\mathrm{comp}}$', r'$\mu_G$', r'$\mu_s$'],
            xticks = self.kkt_eq_offsets[:-1],
            yticklabels= ['stat w', 'stat s', '$H$', 'slacked comp', 'comp $G$', 'comp $s$'],
            yticks = self.kkt_eq_offsets[:-1]
        )
        return

# # initialize a la Waechter2006, Sec. 3.6
# # for initialization
# jac_cost = ca.jacobian(prob.cost, prob.w)
# self.jac_cost_fun = ca.Function('jac_cost_fun', [prob.w], [jac_cost])
# eq_pd = ca.vertcat(H)
# self.grad_H_fun = ca.Function('grad_H_fun', [prob.w], [ca.jacobian(self.H, prob.w)])
# rhs = - np.concatenate((
#     self.jac_cost_fun(prob.w0).full().flatten() - (prob.lbw == 0) * 1,
#     np.zeros(self.n_H)
# ))
# grad_H = self.grad_H_fun(prob.w0).full()
# mat = np.zeros((self.nw + self.n_H, self.nw + self.n_H))
# mat[:self.nw, :self.nw] = np.eye(self.nw)
# mat[:self.nw, self.nw:] = grad_H.T
# mat[self.nw:, :self.nw] = grad_H
# lamH0 = np.linalg.solve(mat, rhs)[self.nw:]
