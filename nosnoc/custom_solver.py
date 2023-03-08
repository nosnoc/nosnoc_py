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
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling, Status
from nosnoc.utils import casadi_vertcat_list, casadi_sum_list, print_casadi_vector, casadi_length, casadi_inf_norm_nan, flatten
from nosnoc.plot_utils import plot_matrix_and_qr, spy_magnitude_plot, spy_magnitude_plot_with_sign

DEBUG = False


def get_fraction_to_boundary(tau: float, current: np.ndarray, delta: np.ndarray, offset: Optional[np.ndarray]=None):
    """Get fraction to boundary.
    :param tau: fraction that should be kept from boundary
    :param current: current value
    :param delta: derivative value
    :param offset: offset for affine function
    """
    if offset is not None:
        delta = delta - offset
    ix = np.where(delta<0)
    if len(ix[0]) == 0:
        return 1.0
    else:
        return min(np.min(-tau *current[ix]/delta[ix]), 1.0)

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
        JIT = False
        casadi_function_opts = {}
        if JIT:
            casadi_function_opts = {"compiler": "shell", "jit": True, "jit_options": {"compiler": "gcc", "flags": ["-O3"]}}

        self.kkt_eq_jac_fun = ca.Function('kkt_eq_jac_fun', [w_pd, prob.p], [kkt_eq, kkt_eq_jac], casadi_function_opts)
        self.kkt_eq_fun = ca.Function('kkt_eq_fun', [w_pd, prob.p], [kkt_eq], casadi_function_opts)
        self.G_fun = ca.Function('G_fun', [w_pd], [self.G], casadi_function_opts)
        self.slack0_fun = ca.Function('slack0_fun', [prob.w, prob.p], [self.slack0_expr], casadi_function_opts)

        # precompute
        self.G_offset = self.G_fun(np.zeros((self.nw_pd,))).full().flatten()

        self.n_comp = n_comp
        self.n_H = n_H
        kkt_block_sizes = [self.nw, n_comp, n_H, n_comp, nG, n_comp]
        self.kkt_eq_offsets = [0]  + np.cumsum(kkt_block_sizes).tolist()

        print(f"created primal dual problem with {casadi_length(w_pd)} variables and {casadi_length(kkt_eq)} equations, {n_comp=}, {self.nw=}, {n_H=}")

        return

    def print_iterate_threshold(self, iterate, threshold=1.0):
        for ii in range(self.nw_pd):
            if np.abs(iterate[ii]) > threshold:
                print(f"{ii}\t{self.w_pd[ii].name():15}\t{iterate[ii]:.2e}")

    def print_iterates(self, iterate_list: list):
        for ii in range(self.nw_pd):
            line = f"{ii}\t{self.w_pd[ii].name():17}"
            for it in iterate_list:
                line += f'\t{it[ii]:.2e}'
            print(line)

    def compute_step_sparse(self, matrix, rhs):
        # naive
        step = scipy.sparse.linalg.spsolve(matrix, rhs)
        return step

    def compute_step_sparse_schur(self, matrix, rhs):
        # Schur complement:
        # [ A, B,
        # C, D]
        # * [x_1, x_2] = [y_1, y_2]
        # ->
        # with S = (A - BD^{-1}C):
        # x1 = S^{-1}(y_1 - B D^{-1}y_2)
        # x2 = D^-1 (y2 - C x_1)
        C = matrix[-self.n_mu:, :-self.n_mu]
        # A = matrix[:-self.n_mu, :-self.n_mu]
        B = matrix[:-self.n_mu, -self.n_mu:]
        y1 = rhs[:-self.n_mu]
        y2 = rhs[-self.n_mu:]
        d_diag_inv = 1/matrix[-self.n_mu:, -self.n_mu:].diagonal()

        D_inv = scipy.sparse.diags(1.0 / matrix[-self.n_mu:, -self.n_mu:].diagonal())
        S = matrix[:-self.n_mu, :-self.n_mu] - B @ D_inv @ C
        x1_rhs = y1 - B @ (y2 * d_diag_inv)
        x1 = scipy.sparse.linalg.spsolve(S, x1_rhs)

        x2 = (y2 - C@x1) * d_diag_inv
        step = np.concatenate((x1, x2))
        return step


    def check_newton_matrix(self, matrix):
        upper_left = matrix[:self.nw, :self.nw]
        lower_right = matrix[-self.n_mu:, -self.n_mu:]
        print(f"conditioning: upper left {np.linalg.cond(upper_left):.3e}, lower right {np.linalg.cond(lower_right):.3e}")
        return

    def get_mu(self, w_current):
        return w_current[-self.n_mu:]

    def get_lambda_comp(self, w_current):
        return w_current[self.nw + self.n_comp + self.n_H: self.nw + 2*self.n_comp+self.n_H]

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
        w_pd_0 = np.concatenate((prob.w0, slack0, lampd0, self.mu_pd_0))

        w_current = w_pd_0

        w_all = [w_current.copy()]
        n_iter_polish = opts.max_iter_homotopy + 1
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]
        sigma_k = opts.sigma_0

        # TODO: make this options
        max_gn_iter = 40
        # line search
        tau_min = .99 # .99 is IPOPT default
        rho = 0.9 # factor to shrink alpha in line search
        gamma = .3
        alpha_min = 0.05

        # timers
        t_la = 0.0
        t_ls = 0.0
        t_ca = 0.0

        n_mu = self.n_mu
        # TODO: initialize duals ala Waechter2006, Sec. 3.6
        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
        if opts.print_level == 1:
            print(f"sigma\t\t iter \t res \t min_steps \t min_mu \t max mu \t min G\t\tmin_lam_comp")

        w_candidate = w_current.copy()
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = sigma_k
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([sigma_k, tau_val]), lambda00, x0))

            if opts.print_level > 1:
                print("alpha \t alpha_mu \t step norm \t kkt res \t min mu \t min G\t min_lam_comp")
            t = time.time()
            alpha_min_counter = 0

            for gn_iter in range(max_gn_iter):

                t0_ca = time.time()
                kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, p_val)
                t_ca += time.time() - t0_ca

                nlp_res = ca.norm_inf(kkt_val).full()[0][0]
                if nlp_res < sigma_k / 10:
                    break

                newton_matrix = jac_kkt_val.sparse()
                # newton_matrix[-n_mu:, -n_mu:] += 1e-5 * scipy.sparse.eye(n_mu)
                # print(f"cond(A) = {np.linalg.cond(newton_matrix.toarray()):.2e}")
                rhs = - kkt_val.full().flatten()

                # cond = np.linalg.cond(newton_matrix.toarray())
                # self.plot_newton_matrix(newton_matrix.toarray(), title=f'Newton matrix, cond() = {cond:.2e}', fig_filename=f'newton_matrix_{ii}_{gn_iter}.pdf')

                t0_la = time.time()
                step = self.compute_step_sparse(newton_matrix, rhs)
                # step = self.compute_step_sparse_schur(newton_matrix, rhs)
                t_la += time.time() - t0_la

                step_norm = np.max(np.abs(step))
                if step_norm < sigma_k:
                    break

                t0_ls = time.time()

                ## LINE SEARCH + fraction to boundary
                tau_j = max(tau_min, 1-tau_val)
                # compute alpha_mu_k
                alpha_mu = get_fraction_to_boundary(tau_j, w_current[-n_mu:], step[-n_mu:], offset=None)
                w_candidate[-n_mu:] = w_current[-n_mu:] + alpha_mu * step[-n_mu:]

                # compute new nlp residual after mu step
                # NOTE: not really necessary, maybe make optional
                # kkt_val = self.kkt_eq_fun(w_candidate, p_val)
                # nlp_res = ca.norm_inf(kkt_val).full()[0][0]

                # fraction to boundary G, s > 0
                G_val = self.G_fun(w_current).full().flatten()
                G_delta_val = self.G_fun(step).full().flatten()
                alpha_max = get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)

                # fraction to boundary lambda_comp
                # alpha_max_lam_comp = get_fraction_to_boundary(tau_j, self.get_lambda_comp(w_current), self.get_lambda_comp(step), offset=None)
                # alpha_max = min(alpha_max, alpha_max_lam_comp)

                # line search:
                alpha = alpha_max
                while True:
                    w_candidate[:-n_mu] = w_current[:-n_mu] + alpha * step[:-n_mu]
                    step_res_norm = ca.norm_inf(self.kkt_eq_fun(w_candidate, p_val)).full()[0][0]
                    if step_res_norm < (1-gamma*alpha) * nlp_res:
                        break
                    elif alpha < alpha_min:
                        alpha_min_counter += 1
                        break
                    else:
                        alpha *= rho

                # do step!
                w_current = w_candidate.copy()
                t_ls += time.time() - t0_ls

                if DEBUG:
                    maxA = np.max(np.abs(newton_matrix))
                    cond = np.linalg.cond(newton_matrix)
                    tmp = newton_matrix.copy()
                    tmp[tmp==0] = 1
                    minA = np.min(np.abs(tmp))
                    print(f"{alpha:.3f} \t {alpha_mu:.3f} \t {step_norm:.2e} \t {nlp_res:.2e} \t {cond:.2e} \t {maxA:.2e} \t {minA:.2e}")
                elif opts.print_level > 1:
                    min_mu = np.min(self.get_mu(w_current))
                    max_mu = np.max(self.get_mu(w_current))
                    min_lam_comp = np.min(self.get_lambda_comp(w_current))
                    print(f"{alpha:.3f} \t {alpha_mu:.3f} \t\t {step_norm:.2e} \t {nlp_res:.2e} \t {min_mu:.2e}\t {np.min(G_val):.2e}\t {min_lam_comp:.2e}")

            # NOTE: tried resetting mu to 1e-4 if it is too small, but this does not help
            # min_mu = np.min(self.get_mu(w_current))
            # if ii < opts.max_iter_homotopy - 1:
            #     w_current[-n_mu:] = np.maximum(w_current[-n_mu:], 1e-4)
            # min_mu_new = np.min(self.get_mu(w_current))
            # print(f"min_mu = {min_mu:.2e}, min_mu_new = {min_mu_new:.2e}")

            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            nlp_iter[ii] = gn_iter
            w_all.append(w_current)

            if opts.print_level > 1:
                print(f"sigma = {sigma_k:.2e}, iter {gn_iter}, res {nlp_res:.2e}, min_steps {alpha_min_counter}")
            elif opts.print_level == 1:
                min_mu = np.min(self.get_mu(w_current))
                max_mu = np.max(self.get_mu(w_current))
                min_lam_comp = np.min(self.get_lambda_comp(w_current))
                print(f"{sigma_k:.2e} \t {gn_iter} \t {nlp_res:.2e} \t {alpha_min_counter}\t {min_mu:.2e} \t {max_mu:.2e} \t {np.min(G_val):.2e}\t{min_lam_comp:.2e}")

            # complementarity_residual = prob.comp_res(w_current[:self.nw], p_val).full()[0][0]
            # complementarity_stats[ii] = complementarity_residual
            # if complementarity_residual < opts.comp_tol:
            #     break

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
        print(f"total iterations {sum_iter}, CPU time {total_time:.3f}: LA: {t_la:.3f} line search: {t_ls:.3f} casadi: {t_ca:.3f} subtimers {t_la+t_ls+t_ca:.3f}")

        if nlp_res < sigma_k:
            results["status"] = Status.SUCCESS
        else:
            results["status"] = Status.NOT_CONVERGED
            condA = np.linalg.cond((newton_matrix.toarray()))
            print(f"did not converge with last condition number {condA:.2e}")
            np.argmax(kkt_val)
            self.print_iterates([w_current, step])
            breakpoint()
            # TODO: check:
            #  w_current[-(self.n_mu+self.n_comp):-self.n_mu]

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
