from typing import Optional, List
import time
import scipy

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
    """
    Get fraction to boundary.
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
        G_list = []
        for i in range(prob.lbw.size):
            if not np.isinf(prob.lbw[i]):
                G_list.append(prob.w[i] - prob.lbw[i])
            if not np.isinf(prob.ubw[i]):
                G_list.append(prob.ubw[i] - prob.w[i])
        G_expr = casadi_vertcat_list(G_list)
        return G_expr

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        super().__init__(opts, model, ocp)
        prob = self.problem

        # complements
        G1 = casadi_vertcat_list([casadi_sum_list(x[0]) for x in prob.comp_list])
        G2 = casadi_vertcat_list([x[1] for x in prob.comp_list])
        # equalities
        self.H = casadi_vertcat_list( [prob.g[i] for i in range(len(prob.lbg)) if prob.lbg[i] == prob.ubg[i]] )

        # G = (G1, G2, s) \geq 0
        self.G_no_slack = self._setup_G()
        nG = casadi_length(self.G_no_slack)

        self.G1 = G1
        self.G2 = G2
        n_comp = casadi_length(G1)
        n_H = casadi_length(self.H)

        # setup primal dual variables:
        slack = ca.SX.sym('slack', n_comp)

        lam_H = ca.SX.sym('lam_H', n_H)
        # lam_comp = ca.SX.sym('lam_comp', n_comp)
        # lam_pd = ca.vertcat(lam_H, lam_comp)
        lam_pd = ca.vertcat(lam_H)
        self.lam_pd_0 = np.zeros((casadi_length(lam_pd),))

        mu_G = ca.SX.sym('mu_G', nG)
        mu_s = ca.SX.sym('mu_s', n_comp)
        mu_pd = ca.vertcat(mu_G, mu_s)
        self.n_mu = casadi_length(mu_pd)

        w_pd = ca.vertcat(prob.w, lam_pd, slack, mu_pd)

        ## KKT system
        self.G = ca.vertcat(self.G_no_slack, slack)

        # slack = - G1 * G2 + sigma
        self.slack0_expr = -ca.diag(G1) @ G2 + prob.sigma

        # barrier parameter
        tau = prob.tau

        # collect KKT system
        slacked_complementarity = slack - self.slack0_expr
        stationarity_w = ca.jacobian(prob.cost, prob.w).T \
            + ca.jacobian(self.H, prob.w).T @ lam_H \
            + ca.jacobian(slacked_complementarity, prob.w).T @ mu_s \
            - ca.jacobian(self.G, prob.w).T @ mu_pd

        kkt_eq_without_comp = ca.vertcat(stationarity_w, self.H, slacked_complementarity)

        # treat IP complementarities:
        kkt_comp = []
        for i in range(nG):
            kkt_comp = ca.vertcat(kkt_comp, self.G_no_slack[i] * mu_G[i] - tau)
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

        # dimensions
        self.nw = casadi_length(prob.w)
        self.nw_pd = casadi_length(w_pd)
        self.n_comp = n_comp
        self.n_H = n_H
        self.nG = nG
        self.n_all_but_mu = self.nw_pd - self.n_mu

        self.kkt_eq_jac_fun = ca.Function('kkt_eq_jac_fun', [w_pd, prob.p], [kkt_eq, kkt_eq_jac], casadi_function_opts)
        self.kkt_eq_fun = ca.Function('kkt_eq_fun', [w_pd, prob.p], [kkt_eq], casadi_function_opts)
        self.G_fun = ca.Function('G_fun', [w_pd], [self.G], casadi_function_opts)
        self.H_fun = ca.Function('H_fun', [w_pd, prob.p], [self.H], casadi_function_opts)
        self.G_no_slack_fun = ca.Function('G_no_slack_fun', [w_pd], [self.G_no_slack], casadi_function_opts)
        self.slack0_fun = ca.Function('slack0_fun', [prob.w, prob.p], [self.slack0_expr], casadi_function_opts)

        self.kkt_max_res_fun = ca.Function('kkt_max_res_fun', [w_pd, prob.p], [ca.norm_inf(kkt_eq)], casadi_function_opts)
        self.max_violation_fun = ca.Function('max_violation_fun', [w_pd, prob.p], [ca.norm_inf(ca.vertcat(slacked_complementarity, self.H))], casadi_function_opts)
        log_merit = prob.cost - tau * ca.sum1(ca.log(self.G))
        self.log_merit_fun = ca.Function('log_merit_fun', [w_pd, prob.p], [log_merit], casadi_function_opts)
        self.log_merit_fun_jac = ca.Function('log_merit_fun_jac', [w_pd, prob.p], [log_merit, ca.jacobian(log_merit, prob.w)], casadi_function_opts)

        # precompute nabla_w_G
        dummy = ca.SX.sym('dummy')
        nabla_w_G_fun = ca.Function('nabla_w_G_fun', [dummy], [ca.jacobian(self.G_no_slack, prob.w).T], casadi_function_opts)
        nabla_w_G = nabla_w_G_fun(1)
        self.nabla_w_G = nabla_w_G_fun(1).sparse()

        # super dense system
        compl_expr = ca.diag(G1) @ G2
        nabla_w_compl = ca.jacobian(compl_expr, prob.w).T
        nabla_w_H = ca.jacobian(self.H, prob.w).T
        mat_elim_mus = ca.SX.zeros((self.nw+n_H, self.nw+n_H))

        GGN_style_hess = False
        if GGN_style_hess:
            stationarity_w_no_H = ca.jacobian(prob.cost, prob.w).T \
                + ca.jacobian(self.H, prob.w).T @ lam_H \
                + ca.jacobian(slacked_complementarity, prob.w).T @ mu_s \
                - ca.jacobian(self.G, prob.w).T @ mu_pd

            mat_elim_mus[:self.nw, :self.nw] = ca.jacobian(stationarity_w_no_H, prob.w) + \
                nabla_w_G @ ca.diag(mu_G/self.G_no_slack) @ nabla_w_G.T + nabla_w_compl @ ca.diag(mu_s/ slack) @ nabla_w_compl.T
        else:
            mat_elim_mus[:self.nw, :self.nw] = ca.jacobian(stationarity_w, prob.w) + \
                nabla_w_G @ ca.diag(mu_G/self.G_no_slack) @ nabla_w_G.T + nabla_w_compl @ ca.diag(mu_s/ slack) @ nabla_w_compl.T
        mat_elim_mus[:self.nw, self.nw:] = nabla_w_H
        mat_elim_mus[self.nw:, :self.nw] = nabla_w_H.T

        r_lw_dtilde = stationarity_w \
                    + nabla_w_G @ ca.diag(1/self.G_no_slack) @ kkt_comp[:nG] \
                    - nabla_w_compl @ ca.diag(1/slack) @ kkt_comp[nG:] \
                    + nabla_w_compl @ ca.diag(mu_s / slack) @ slacked_complementarity

        self.dense_ls_fun = ca.Function('dense_ls_fun', [w_pd, prob.p], [mat_elim_mus, r_lw_dtilde, nabla_w_compl])
        self.nabla_w_compl_fun = ca.Function('nabla_w_compl_fun', [w_pd, prob.p], [nabla_w_compl])
        self.slacked_compl_fun = ca.Function('slacked_compl_fun', [w_pd, prob.p], [slacked_complementarity])

        # precompute
        self.G_offset = self.G_fun(np.zeros((self.nw_pd,))).full().flatten()

        self.w_pd = w_pd

        kkt_block_sizes = [self.nw, n_H, n_comp, nG, n_comp]
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

    def print_G_val(self, G_val):
        for ii in range(casadi_length(self.G)):
            eq_str = str(self.G[ii])
            print(f"{ii}\t{eq_str:17}\t{G_val[ii]:.2e}")

    def print_kkt_residual(self, kkt_res):
        for ii in range(self.nw_pd):
            eq_str = str(self.kkt_eq[ii])
            print(f"{eq_str:20}\t{kkt_res[ii]:.2e}")

    def compute_step_sparse(self, matrix, rhs):
        # naive
        step = scipy.sparse.linalg.spsolve(matrix, rhs)
        return step

    def compute_step_elim_mu_s(self, w_current):
        mat, r_lw_tilde, nabla_w_compl = self.dense_ls_fun(w_current, self.p_val)
        mu_G = w_current[-self.n_mu:-self.n_comp]

        r_G = -self.kkt_val[-self.n_mu:-self.n_comp]
        r_s = -self.kkt_val[-self.n_comp:]
        r_comp = -self.kkt_val[self.nw+self.n_H: self.nw+self.n_H+self.n_comp]
        r_eq = -self.kkt_val[self.nw : self.nw+self.n_H]

        G_val = self.G_no_slack_fun(w_current).full().flatten()
        self.r_lw_tilde = -r_lw_tilde.full().flatten()

        rhs_elim = np.concatenate((self.r_lw_tilde, r_eq))

        # from nosnoc.plot_utils import latexify_plot
        # latexify_plot()
        # spy_magnitude_plot(mat.toarray())
        # import matplotlib.pyplot as plt
        # plt.show()

        SPARSE = True
        if SPARSE:
            self.mat = mat.sparse()
            # step_w_lam = scipy.sparse.linalg.spsolve(mat, rhs_elim)
            self.lu_factor = scipy.sparse.linalg.factorized(self.mat)
            step_w_lam = self.lu_factor(rhs_elim)

            # mat = mat.full()
            # L, D, perm = scipy.linalg.ldl(mat, lower=True)
            # y = np.linalg.solve(L, rhs_elim[perm])
            # x = np.linalg.solve(D, y)
            # step_w_lam = np.zeros_like(x)
            # step_w_lam[perm] = x
        else:
            mat = mat.full()
            # step_w_lam = np.linalg.solve(mat, rhs_elim)
            step_w_lam = scipy.linalg.solve(mat, rhs_elim)

        slack_current = w_current[self.nw+self.n_H: self.nw+self.n_H+self.n_comp]

        # expand
        delta_slack = r_comp - nabla_w_compl.T.full() @ step_w_lam[:self.nw]
        delta_mu_G = (r_G - mu_G * (self.nabla_w_G.T @ step_w_lam[:self.nw])) / G_val
        delta_mu_s = (r_s - delta_slack * w_current[-self.n_comp:]) / slack_current

        step = np.concatenate((step_w_lam, delta_slack, delta_mu_G, delta_mu_s))
        return step

    def get_mu(self, w_current):
        return w_current[-self.n_mu:]

    def get_slack(self, w_current):
        return w_current[self.nw+self.n_H:self.nw+self.n_H+self.n_comp]

    def get_second_order_correction(self, w_current, w_candidate, alpha):
        mu_G = w_current[-self.n_mu:-self.n_comp]
        # setup rhs
        r_eq_candidate = self.H_fun(w_candidate, self.p_val).full().flatten()
        r_eq_soc = -(alpha * self.kkt_val[self.nw : self.nw+self.n_H] + r_eq_candidate)

        # this one is a residual and should be evaluated again
        r_comp_cand = self.slacked_compl_fun(w_candidate, self.p_val).full().flatten()
        r_comp_soc = - (alpha * self.kkt_val[self.nw+self.n_H: self.nw+self.n_H+self.n_comp] + r_comp_cand)

        A_k = self.mat[:self.nw, self.nw:]
        rhs_elim = np.concatenate((self.r_lw_tilde -A_k @ w_current[self.nw:self.nw+self.n_H], r_eq_soc))

        # compute step
        step_w_lam = self.lu_factor(rhs_elim)
        # this is an equality jacobian and should not be evaluated again (?)
        nabla_w_compl = self.nabla_w_compl_fun(w_current, self.p_val)
        delta_slack = r_comp_soc - nabla_w_compl.T.full() @ step_w_lam[:self.nw]

        # expand mu
        kkt_val = self.kkt_eq_fun(w_candidate, self.p_val).full().flatten()
        G_val = self.G_no_slack_fun(w_current).full().flatten()
        slack_current = w_current[self.nw+self.n_H: self.nw+self.n_H+self.n_comp]

        r_G = -kkt_val[-self.n_mu:-self.n_comp]
        r_s = -kkt_val[-self.n_comp:]
        delta_mu_G = (r_G - mu_G * (self.nabla_w_G.T @ step_w_lam[:self.nw])) / G_val
        delta_mu_s = (r_s - delta_slack * w_current[-self.n_comp:]) / slack_current

        step = np.concatenate((step_w_lam, delta_slack, delta_mu_G, delta_mu_s))

        return step


    def check_Waechter20(self, step_log_merit, eta_phi, step, alpha, dir_der_log_merit) -> bool:
        return (step_log_merit < self.log_merit + eta_phi * alpha * dir_der_log_merit)


    def check_Waechter19(self, step, alpha, theta_current, dir_der_log_merit) -> bool:
        s_phi = 2.3 # IPOPT: 2.3
        s_theta = 1.1 # IPOPT: 1.1
        delta = 1.0 # IPOPT: 1.0

        if dir_der_log_merit >= 0.0:
            return False
        if alpha * (-dir_der_log_merit)**s_phi > delta * theta_current**s_theta:
            return True
        else:
            return False

    def check_Waechter18(self, step_viol, theta_current, step_log_merit):
        gamma_phi = 1e-8
        gamma_theta = 1e-5
        if step_viol < (1-gamma_theta) * theta_current:
            return True
        elif step_log_merit < self.log_merit - gamma_phi*theta_current:
            return True
        else:
            return False



    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()

        tau_val = opts.sigma_0
        sigma_k = opts.sigma_0

        self.setup_p_val(sigma_k, tau_val)

        lamH0 = 1.0 * np.ones((self.n_H,))
        mu_pd_0 = np.ones((self.n_mu,))
        # slack0 = self.slack0_fun(prob.w0, self.p_val).full().flatten()
        slack0 = np.ones((self.n_comp,))
        w_current = np.concatenate((prob.w0, lamH0, slack0, mu_pd_0))

        w_all = [w_current.copy()]
        n_max_iter_inc_polish = opts.max_iter_homotopy + 1
        complementarity_stats = n_max_iter_inc_polish * [None]
        cpu_time_nlp = n_max_iter_inc_polish * [None]
        nlp_iter = n_max_iter_inc_polish * [None]

        # TODO: make this options
        max_newton_iter = 100
        # line search
        tau_min = .99 # .99 is IPOPT default
        rho = 0.9 # factor to shrink alpha in line search
        alpha_min = 0.01
        kappaSigma = 1e10 # 1e10 is IPOPT default
        theta_min_fact = 0.0001 # IPOPT 0.0001

        # timers
        t_la = 0.0
        t_ls = 0.0
        t_ca = 0.0

        n_mu = self.n_mu
        # TODO: initialize duals ala Waechter2006, Sec. 3.6
        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
        if opts.print_level == 1:
            print(f"sigma\t\t iter \t res \t\tmin_steps \t min_mu \t min G")

        w_candidate = w_current.copy()
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = sigma_k
            self.setup_p_val(sigma_k, tau_val)

            if opts.print_level > 1:
                print("alpha \t alpha_mu \talpha_max\t step norm \t kkt res \t min mu \t min G")
            t = time.time()
            self.alpha_min_counter = 0

            theta_0 = self.max_violation_fun(w_current, self.p_val).full()[0][0]

            for newton_iter in range(max_newton_iter):

                t0_ca = time.time()
                # kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, self.p_val)
                # self.kkt_val = kkt_val.full().flatten()
                kkt_val = self.kkt_eq_fun(w_current, self.p_val)
                self.kkt_val = kkt_val.full().flatten()
                t_ca += time.time() - t0_ca

                nlp_res = ca.norm_inf(kkt_val).full()[0][0]
                if nlp_res < sigma_k / 10:
                    break

                # simple sparse
                # newton_matrix = jac_kkt_val.sparse()
                # rhs = - kkt_val.full().flatten()
                # step = self.compute_step_sparse(newton_matrix, rhs)

                # cond = np.linalg.cond(newton_matrix.toarray())
                # self.plot_newton_matrix(newton_matrix.toarray(), title=f'Newton matrix, cond() = {cond:.2e}', fig_filename=f'newton_matrix_{ii}_{newton_iter}.pdf')

                t0_la = time.time()
                step = self.compute_step_elim_mu_s(w_current)
                t_la += time.time() - t0_la

                step_norm = np.max(np.abs(step))
                if step_norm < sigma_k / 10:
                    break

                t0_ls = time.time()
                theta_current = self.max_violation_fun(w_current, self.p_val).full()[0][0]

                # log_merit = self.log_merit_fun(w_current, self.p_val).full()[0][0]
                log_merit, dlog_merit_x = self.log_merit_fun_jac(w_candidate, self.p_val)
                self.log_merit = log_merit.full()[0][0]
                self.dlog_merit_x = dlog_merit_x.full()

                ## LINE SEARCH + fraction to boundary
                tau_j = max(tau_min, 1-tau_val)

                # compute new nlp residual after mu step
                # NOTE: not really necessary, maybe make optional
                # kkt_val = self.kkt_eq_fun(w_candidate, self.p_val)
                # nlp_res = ca.norm_inf(kkt_val).full()[0][0]

                # fraction to boundary G, s > 0
                G_val = self.G_fun(w_current).full().flatten()
                G_delta_val = self.G_fun(step).full().flatten()
                alpha_max = get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)
                # alpha_max_slack = get_fraction_to_boundary(tau_j, self.get_slack(w_current), self.get_slack(step))

                if any(G_val < 0):
                    print("G_val < 0 should never happen")
                    breakpoint()
                eta_phi = 1e-8

                # line search:
                alpha = alpha_max
                soc_iter = False
                theta_min = theta_min_fact * max(1, theta_0)
                while True:
                    if alpha < alpha_min:
                        self.alpha_min_counter += 1
                        # condA = np.linalg.cond(self.mat.toarray())
                        # print(f"got min step at {ii} {newton_iter}, with alpha = {alpha:.2e}, {alpha_max_slack:.2e} kkt res: {nlp_res:.2e} cond: {condA:.2e}")
                        # breakpoint()
                        break
                    w_candidate[:self.n_all_but_mu] = w_current[:self.n_all_but_mu] + alpha * step[:self.n_all_but_mu]
                    # t0_ca = time.time()
                    step_viol = self.max_violation_fun(w_candidate, self.p_val).full()[0][0]

                    step_log_merit = self.log_merit_fun(w_candidate, self.p_val).full()[0][0]
                    dir_der_log_merit = self.dlog_merit_x @ step[:self.nw]

                    if not self.check_Waechter19(step, alpha, theta_current, dir_der_log_merit):
                        # Waechter case 2
                        if self.check_Waechter18(step_viol, theta_current, step_log_merit):
                            break
                    elif theta_current < theta_min:
                        # Waechter case 1
                        if self.check_Waechter20(step_log_merit, eta_phi, step, alpha, dir_der_log_merit):
                            break
                    else:
                        # Waechter case 2
                        if self.check_Waechter18(step_viol, theta_current, step_log_merit):
                            break
                    # reject step
                    if alpha == alpha_max: # if this is the first step
                        # if True:
                        step = self.get_second_order_correction(w_current, w_candidate, alpha)
                        soc_iter = True
                        G_delta_val = self.G_fun(step).full().flatten()
                        alpha = get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)
                        print(f"SOC at it {ii} {newton_iter}. alpha_max {alpha:.2e} theta_step {step_viol:.2e} theta {theta_current:.2e} delta phi {step_log_merit:.2e} phi {self.log_merit:.2e}")
                    alpha *= rho

                if soc_iter:
                    print(f"alpha {alpha:.2e} theta_step {step_viol:.2e} theta {theta_current:.2e} delta phi {step_log_merit:.2e} phi {self.log_merit:.2e}")

                # compute alpha_mu_k
                alpha_mu = get_fraction_to_boundary(tau_j, w_current[-n_mu:], step[-n_mu:], offset=None)
                w_candidate[-n_mu:] = w_current[-n_mu:] + alpha_mu * step[-n_mu:]

                # do step
                w_current = w_candidate.copy()
                # Waechter2006 eq (16): "Primal-dual barrier term Hessian should not deviate arbitrarily much from primal Hessian"
                G_val = self.G_fun(w_current).full().flatten()
                for i in range(n_mu):
                    new = max(min(w_current[-n_mu+i], kappaSigma * tau_val / G_val[i]), tau_val/(kappaSigma * G_val[i]))
                    # if new != w_current[-n_mu+i]:
                    #     print(f"mu[{i}] = {w_current[-n_mu+i]} -> {new}")
                    w_current[-n_mu+i] = new
                t_ls += time.time() - t0_ls

                if opts.print_level > 1:
                    min_mu = np.min(self.get_mu(w_current))
                    max_mu = np.max(self.get_mu(w_current))
                    max_slack_comp_viol = np.max(np.abs(kkt_val[self.nw+self.n_H:self.nw+self.n_H+self.n_comp]))
                    max_H_viol = np.max(np.abs(kkt_val[self.nw:self.nw+self.n_H]))
                    max_comp_viol = np.max(np.abs(kkt_val[:-self.n_mu]))
                    # min_lam_comp = np.min(self.get_lambda_comp(w_current))
                    print(f"{alpha:.3f} \t {alpha_mu:.3f} \t\t {alpha_max:.3f} \t\t {step_norm:.2e} \t {nlp_res:.2e} \t {min_mu:.2e}\t {np.min(G_val):.2e}\t {max_slack_comp_viol:.2e}\t{max_H_viol:.2e}\t{max_comp_viol:.2e}")

            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            nlp_iter[ii] = newton_iter
            w_all.append(w_current)

            if opts.print_level > 1:
                print(f"sigma = {sigma_k:.2e}, iter {newton_iter}, res {nlp_res:.2e}, min_steps {self.alpha_min_counter}")
            elif opts.print_level == 1:
                min_mu = np.min(self.get_mu(w_current))
                max_mu = np.max(self.get_mu(w_current))
                # min_lam_comp = np.min(self.get_lambda_comp(w_current))
                print(f"{sigma_k:.2e} \t {newton_iter} \t {nlp_res:.2e} \t {self.alpha_min_counter}\t\t {min_mu:.2e} \t {np.min(G_val):.2e}\t")

            # complementarity_residual = prob.comp_res(w_current[:self.nw], self.p_val).full()[0][0]
            # complementarity_stats[ii] = complementarity_residual
            # if complementarity_residual < opts.comp_tol:
            #     break

            # Update the homotopy parameter.
            sigma_k = self.homotopy_sigma_update(sigma_k)

        # if opts.do_polishing_step: ...

        # collect results
        results = get_results_from_primal_vector(prob, w_current)

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_WOPT_PREV:
            prob.w0[:] = w_current[:self.nw]
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_current
        results["cost_val"] = 0.0

        sum_iter = sum([i for i in nlp_iter if i is not None])
        total_time = sum([i for i in cpu_time_nlp if i is not None])
        print(f"total iterations {sum_iter}, CPU time {total_time:.3f}: LA: {t_la:.3f} line search: {t_ls:.3f} casadi: {t_ca:.3f} subtimers {t_la+t_ls+t_ca:.3f}")

        if nlp_res < sigma_k or step_norm < sigma_k:
            results["status"] = Status.SUCCESS
        else:
            results["status"] = Status.NOT_CONVERGED
            ikkt = np.argmax(kkt_val)
            condA = np.linalg.cond(self.mat.toarray())
            print(f"biggest KKT res at {ikkt} with value {self.kkt_eq[ikkt]} = {kkt_val.full()[ikkt][0]:.2e}, condA = {condA:.2e}")
            print("continue to print iterate and step")
            breakpoint()
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
            xticklabels=[r'$w$', r'$\lambda_H$', r'$s$', r'$\mu_G$', r'$\mu_s$'],
            xticks = self.kkt_eq_offsets[:-1],
            yticklabels= ['stat w', '$H$', 'slacked comp', 'comp $G$', 'comp $s$'],
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
