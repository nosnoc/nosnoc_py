from dataclasses import dataclass
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

class NosnocFilter():
    # theta, phi
    def __init__(self, theta_max: float, gamma_theta: float, m_max: int = 0) -> None:
        self.theta_max = theta_max
        self.gamma_theta = gamma_theta
        self.filter_points = set()
        self.m_max = m_max
        return

    def add(self, point: tuple) -> None:
        self.filter_points.add(point)

    def is_acceptable(self, point: tuple) -> bool:
        # checks if point \notin filter
        theta, phi = point
        if theta > self.theta_max:
            return False
        #
        count = 0
        for filter_point in self.filter_points:
            theta_k, phi_k = filter_point
            if theta >= (1-self.gamma_theta) * theta_k and phi >= phi_k - self.gamma_theta * theta_k:
                count += 1
                if count > self.m_max:
                    return False
        return True

@dataclass
class NosnocCustomSolverOpts:
    max_newton_iter: int = 100
    kappa_res_sigma: float = 5.0 # break loop if nlp_res < kappa_res_sigma * sigma, IPOPT-C: \delta_\mu -- default 5.0
    # line search
    tau_min: float = .95 # .99 is IPOPT default
    rho: float = 0.5 # factor to shrink alpha in line search # IPOPT: 0.5
    kappaSigma: float = 1e10 # 1e10 is IPOPT default
    theta_min_fact: float = 0.0001 # IPOPT 0.0001
    max_soc: int = 4 # IPOPT 4
    kappa_soc: float = 0.99 # IPOPT: 0.99
    gamma_theta: float = 1e-5 # IPOPT gamma_theta 1e-5
    gamma_phi: float = 1e-8 # IPOPT: 1e-8
    gamma_alpha: float = 0.05 # IPOPT: 0.05
    s_phi: float = 2.3 # IPOPT: 2.3
    s_theta = 1.1 # IPOPT: 1.1
    eta_phi: float = 1e-8
    delta = 1.0 # IPOPT: 1.0

    # restoration phase
    rho_resto = 1e3 # IPOPT: 1e3
    kappa_resto = 0.9 # IPOPT: 0.9

class NosnocCustomSolver(NosnocSolverBase):
    def get_fraction_to_boundary(self, tau: float, current: np.ndarray, delta: np.ndarray, offset: Optional[np.ndarray]=None):
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
            # if len(current) == self.nG:
            #     argmin = np.argmin(-tau *current[ix]/delta[ix])
            #     og_idx = ix[0][argmin]
            #     print(f"fraction_to_boundary, tau = {tau:.5e} idx {og_idx}, {self.G[og_idx]} min {min(np.min(-tau *current[ix]/delta[ix]), 1.0)}")
            return min(np.min(-tau *current[ix]/delta[ix]), 1.0)

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

        # options
        self.solver_opts: NosnocCustomSolverOpts = NosnocCustomSolverOpts()

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
        self.violation_norm_fun = ca.Function('violation_norm_fun', [w_pd, prob.p], [ca.norm_1(ca.vertcat(slacked_complementarity, self.H))], casadi_function_opts)
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

        delta_w = ca.SX.sym('delta_w')
        delta_c = ca.SX.sym('delta_c')

        GGN_style_hess = False

        eps_reg = 1e-15
        # slack_tilde = ca.fmax(slack, eps_reg)
        # G_tilde = ca.fmax(self.G_no_slack, eps_reg)

        slack_tilde = slack
        G_tilde = self.G_no_slack
        # TODO: try diag(mu/G) =: W to bound [1e-8, 1e8]

        if GGN_style_hess:
            stationarity_w_no_H = ca.jacobian(prob.cost, prob.w).T \
                + ca.jacobian(self.H, prob.w).T @ lam_H \
                + ca.jacobian(slacked_complementarity, prob.w).T @ mu_s \
                - ca.jacobian(self.G, prob.w).T @ mu_pd

            mat_elim_mus[:self.nw, :self.nw] = ca.jacobian(stationarity_w_no_H, prob.w) + \
                nabla_w_G @ ca.diag(mu_G/ G_tilde) @ nabla_w_G.T + nabla_w_compl @ ca.diag(mu_s / slack_tilde) @ nabla_w_compl.T
        else:
            mat_elim_mus[:self.nw, :self.nw] = ca.jacobian(stationarity_w, prob.w) + \
                nabla_w_G @ ca.diag(mu_G/ G_tilde) @ nabla_w_G.T + nabla_w_compl @ ca.diag(mu_s / slack_tilde) @ nabla_w_compl.T
        mat_elim_mus[:self.nw, self.nw:] = nabla_w_H
        mat_elim_mus[self.nw:, :self.nw] = nabla_w_H.T

        #
        mat_elim_mus[:self.nw, :self.nw] += delta_w * ca.diag(np.ones(self.nw))
        mat_elim_mus[self.nw:, self.nw:] += - delta_c * ca.diag(np.ones(n_H))

        r_lw_dtilde = stationarity_w \
                    + nabla_w_G @ ca.diag(1/self.G_no_slack) @ kkt_comp[:nG] \
                    - nabla_w_compl @ ca.diag(1/slack) @ kkt_comp[nG:] \
                    + nabla_w_compl @ ca.diag(mu_s / slack) @ slacked_complementarity

        self.dense_ls_fun = ca.Function('dense_ls_fun', [w_pd, prob.p, delta_w, delta_c], [mat_elim_mus, r_lw_dtilde, nabla_w_compl])
        self.nabla_w_compl_fun = ca.Function('nabla_w_compl_fun', [w_pd, prob.p], [nabla_w_compl])
        self.slacked_compl_fun = ca.Function('slacked_compl_fun', [w_pd, prob.p], [slacked_complementarity])

        # precompute
        self.G_offset = self.G_fun(np.zeros((self.nw_pd,))).full().flatten()

        self.w_pd = w_pd

        kkt_block_sizes = [self.nw, n_H, n_comp, nG, n_comp]
        self.kkt_eq_offsets = [0]  + np.cumsum(kkt_block_sizes).tolist()

        print(f"created primal dual problem with {casadi_length(w_pd)} variables and {casadi_length(kkt_eq)} equations, {n_comp=}, {self.nw=}, {n_H=}")

        ## restoration stuff
        resto_c = ca.vertcat(self.H, slacked_complementarity)
        resto_x = ca.vertcat(prob.w, slack)
        n_c = casadi_length(resto_c)
        nx_resto = casadi_length(resto_x)

        # parameter of resto problem
        resto_weights = ca.SX.sym('D_R', nx_resto)
        resto_x_R = ca.SX.sym('resto_x_R', nx_resto)
        zeta = ca.SX.sym('zeta')
        resto_rho = ca.SX.sym('rho')
        resto_param = ca.vertcat(prob.p, zeta, resto_rho, resto_x_R, resto_weights)

        dG_dx_fun = ca.Function('dG_dx_fun', [dummy], [ca.jacobian(self.G, resto_x)])
        self.dG_dx = dG_dx_fun(1).full()

        # variables
        resto_p = ca.SX.sym('resto_p', n_c)
        resto_n = ca.SX.sym('resto_n', n_c)
        resto_np = ca.vertcat(resto_n, resto_p)
        diff_x = resto_x - resto_x_R
        cost_resto = resto_p + resto_n + .5 * zeta * diff_x.T @ ca.diag(resto_weights) @ diff_x # Waechter (30)

        constraints_resto = resto_c - resto_p + resto_n
        # duals:
        lam_comp = ca.SX.sym('lam_comp', n_comp)
        lambda_resto = ca.vertcat(lam_H, lam_comp)
        z_resto = ca.vertcat(mu_G, mu_s)
        z_n = ca.SX.sym('z_n', n_c)
        z_p = ca.SX.sym('z_p', n_c)
        resto_duals = ca.vertcat(lam_H, lam_comp, z_resto, z_n, z_p)

        # linear system
        weighted_constraint_sum = casadi_sum_list([lambda_resto[i] * resto_c[i] for i in range(n_c)])

        D_R_squared = ca.diag(resto_weights * resto_weights)
        W, _ = ca.hessian( weighted_constraint_sum, resto_x)
        Sigma = ca.jacobian(self.G, resto_x).T @ ca.diag(self.G / z_resto) @ ca.jacobian(self.G, resto_x)
        top_left = W + zeta * D_R_squared + Sigma

        nabla_c = ca.jacobian(resto_c, resto_x).T
        Sigma_p = ca.diag(z_p / resto_p)
        Sigma_n = ca.diag(z_n / resto_n)
        Sigma_p_inv = ca.diag(resto_p / z_p)
        Sigma_n_inv = ca.diag(resto_n / z_n)

        mat_resto = ca.blockcat(top_left, nabla_c, nabla_c.T, -Sigma_p_inv-Sigma_n_inv)

        rhs_resto_1 = zeta * D_R_squared @ diff_x + \
                    nabla_c @ lambda_resto - tau * ca.jacobian(self.G, resto_x).T @ (1/self.G)

        rhs_resto_2 = resto_c - resto_p + resto_n + resto_rho * ((tau-resto_p)/z_p) + resto_rho * ((tau-resto_n)/z_n)

        rhs_resto = - ca.vertcat(rhs_resto_1, rhs_resto_2)
        self.ls_resto_fun = ca.Function('ls_resto_fun', [resto_x, resto_np, resto_duals, resto_param], [mat_resto, rhs_resto, self.G])

        self.resto_c_fun = ca.Function('resto_c_fun', [resto_x, resto_param], [resto_c])

        return

    def print_iterate_threshold(self, iterate, threshold=1.0):
        for ii in range(self.nw_pd):
            if np.abs(iterate[ii]) > threshold:
                print(f"{ii}\t{self.w_pd[ii].name():15}\t{iterate[ii]:.2e}")

    def print_iterates(self, iterate_list: list):
        for ii in range(self.nw_pd):
            line = f"{ii}\t{self.w_pd[ii].name():17}"
            for it in iterate_list:
                line += f'\t{it[ii]:.4e}'
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

        delta_w = 0.0
        delta_c = 0.0
        mat, r_lw_tilde, nabla_w_compl = self.dense_ls_fun(w_current, self.p_val, delta_w, delta_c)
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

            cond = np.linalg.cond(self.mat.toarray())
            eigen_np = np.linalg.eigvals(self.mat.toarray()).tolist()

            print(f"cond = {cond:e}")
            # inertia correction
            kappa_w_minus = 1/3
            kappa_w_plus = 8
            kappa_w_plus_bar = 100
            delta_w_min = 1e-20
            kappa_c = 1/4
            delta_c_bar = 1e-8 # IPOPT: 1e-8
            delta_w_max = 1e40
            delta_w_0 = 1e-4

            ic_iter = 0
            while any(np.iscomplex(eigen_np)): #cond > 1e13:
                # update delta_w
                if ic_iter == 0:
                    if self.delta_w_last == 0.0:
                        delta_w = delta_w_0
                    else:
                        delta_w = max(delta_w_min, kappa_w_minus * self.delta_w_last)
                    delta_c = delta_c_bar * self.tau_val**kappa_c
                else:
                    if self.delta_w_last == 0.0:
                        delta_w = kappa_w_plus_bar * delta_w
                    else:
                        delta_w = kappa_w_plus * delta_w

                mat, r_lw_tilde, nabla_w_compl = self.dense_ls_fun(w_current, self.p_val, delta_w, delta_c)

                self.mat = mat.sparse()
                cond = np.linalg.cond(self.mat.toarray())
                eigen_np = np.linalg.eigvals(self.mat.toarray()).tolist()
                print(f"inertia control cond = {cond:e}, delta_w {delta_w:e}, delta_c {delta_c:e}")

                if delta_w > delta_w_max:
                    # TODO: restoration
                    breakpoint()

                if ic_iter > 100:
                    breakpoint()

                if not any(np.iscomplex(eigen_np)):
                    eigen_list = sorted(eigen_np)
                    eps_eig = 1e-5
                    neg_eig = [x for x in eigen_list if x < -eps_eig]
                    pos_eig = [x for x in eigen_list if x > eps_eig]
                    print(f"eigenvalues pos: {len(pos_eig)}, neg: {len(neg_eig)}")
                ic_iter += 1

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

    def get_second_order_correction(self, w_current, w_candidate, alpha, soc_iter):
        mu_G = w_current[-self.n_mu:-self.n_comp]
        mu_s = w_current[-self.n_comp:]

        # setup rhs
        r_eq_candidate = self.H_fun(w_candidate, self.p_val).full().flatten()
        r_comp_cand = self.slacked_compl_fun(w_candidate, self.p_val).full().flatten()
        if soc_iter == 0:
            self.c_soc_k_eq = alpha * self.kkt_val[self.nw : self.nw+self.n_H] + r_eq_candidate
            self.c_soc_k_comp = alpha * self.kkt_val[self.nw+self.n_H: self.nw+self.n_H+self.n_comp] + r_comp_cand
        else:
            self.c_soc_k_eq = alpha * self.c_soc_k_eq + r_eq_candidate
            self.c_soc_k_comp = alpha * self.c_soc_k_comp + r_comp_cand

        A_k = self.mat[:self.nw, self.nw:]
        rhs_elim = np.concatenate((self.r_lw_tilde -A_k @ w_current[self.nw:self.nw+self.n_H], - self.c_soc_k_eq))

        # compute step
        step_w_lam = self.lu_factor(rhs_elim)
        # this is an equality jacobian and should not be evaluated again (?)
        nabla_w_compl = self.nabla_w_compl_fun(w_current, self.p_val)
        delta_slack = - self.c_soc_k_comp - nabla_w_compl.T.full() @ step_w_lam[:self.nw]

        # expand mu
        kkt_val = self.kkt_eq_fun(w_candidate, self.p_val).full().flatten()
        G_val = self.G_no_slack_fun(w_current).full().flatten()
        slack_current = self.get_slack(w_current)

        r_G = -kkt_val[-self.n_mu:-self.n_comp]
        r_s = -kkt_val[-self.n_comp:]
        delta_mu_G = (r_G - mu_G * (self.nabla_w_G.T @ step_w_lam[:self.nw])) / G_val
        delta_mu_s = (r_s - delta_slack * mu_s) / slack_current

        step = np.concatenate((step_w_lam, delta_slack, delta_mu_G, delta_mu_s))

        return step


    def check_Waechter20(self, step_log_merit, step, alpha, dir_der_log_merit) -> bool:
        return (step_log_merit < self.log_merit + self.solver_opts.eta_phi * alpha * dir_der_log_merit)

    def check_Waechter19(self, step, alpha, dir_der_log_merit) -> bool:
        solver_opts = self.solver_opts

        if dir_der_log_merit >= 0.0:
            return False
        if alpha * (-dir_der_log_merit)**solver_opts.s_phi > solver_opts.delta * self.theta_current**solver_opts.s_theta:
            return True
        else:
            return False

    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.
        """
        opts = self.opts
        solver_opts = self.solver_opts
        prob = self.problem

        # initialize
        self.initialize()

        self.tau_val = opts.sigma_0
        sigma_k = opts.sigma_0

        self.setup_p_val(sigma_k, self.tau_val)

        lamH0 = 1.0 * np.ones((self.n_H,))
        mu_pd_0 = np.ones((self.n_mu,))
        # slack0 = self.slack0_fun(prob.w0, self.p_val).full().flatten()
        slack0 = np.ones((self.n_comp,))
        w_current = np.concatenate((prob.w0, lamH0, slack0, mu_pd_0))

        w_all = [w_current.copy()]
        n_iter_polish = opts.max_iter_homotopy + (1 if opts.do_polishing_step else 0)
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]

        self.delta_w_last = 0.0 # inertia correction

        # timers
        t_la = 0.0
        t_ls = 0.0
        t_ca = 0.0

        n_mu = self.n_mu
        # TODO: initialize duals ala Waechter2006, Sec. 3.6
        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
        if opts.print_level == 1:
            print(f"sigma\t\titer \tres \t\tmin_steps\tmin_mu\t\tmin G")

        w_candidate = w_current.copy()
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            # setting tau = sigma seems to be a good choice, also reported in IPOPT-C paper.
            self.tau_val = sigma_k
            self.setup_p_val(sigma_k, self.tau_val)

            if opts.print_level > 1:
                print("alpha\talpha_mu\talpha_max\tstep norm\tkkt res\t\tmin_mu\t\tmin G")
            t = time.time()
            self.alpha_min_counter = 0

            G_val = self.G_fun(w_current).full().flatten()

            # TODO: should this be evaluated every homotopy iteration?
            theta_0 = self.violation_norm_fun(w_current, self.p_val).full()[0][0]
            theta_min = self.solver_opts.theta_min_fact * max(1, theta_0)
            theta_max = 1e4 * max(1, theta_0)

            # initialize filter
            self.filter = NosnocFilter(theta_max=theta_max, gamma_theta=solver_opts.gamma_theta)

            for newton_iter in range(solver_opts.max_newton_iter):

                t0_ca = time.time()
                # kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, self.p_val)
                # self.kkt_val = kkt_val.full().flatten()
                kkt_val = self.kkt_eq_fun(w_current, self.p_val)
                self.kkt_val = kkt_val.full().flatten()
                t_ca += time.time() - t0_ca

                nlp_res = ca.norm_inf(kkt_val).full()[0][0]
                if nlp_res < solver_opts.kappa_res_sigma * sigma_k:
                    break

                # eq (5) Waechter
                s_max = 100
                lam_max = np.max(np.abs(w_current[self.nw:self.nw+self.n_H]))
                mu_max = np.max(self.get_mu(w_current))
                s_d = max(s_max, (lam_max + mu_max) / (self.nw+self.n_H+self.n_comp)) / s_max
                s_c = max(s_max, mu_max/self.nw) / s_max

                scaled_stat = ca.norm_inf(kkt_val[:self.nw]).full()[0][0] / s_d
                scaled_comp = ca.norm_inf(kkt_val[-self.nG-self.n_comp:]).full()[0][0] / s_c
                eq_res = ca.norm_inf(kkt_val[self.nw:self.nw+self.n_H]).full()[0][0]
                print(f"scaled residuals: {scaled_stat:e}, {eq_res:e}, {scaled_comp:e}")

                if max(scaled_comp, eq_res) < solver_opts.kappa_res_sigma * sigma_k:
                # if max(scaled_stat, scaled_comp, eq_res) < solver_opts.kappa_res_sigma * sigma_k:
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

                t0_ls = time.time()
                self.theta_current = self.violation_norm_fun(w_current, self.p_val).full()[0][0]

                # log_merit = self.log_merit_fun(w_current, self.p_val).full()[0][0]
                log_merit, dlog_merit_x = self.log_merit_fun_jac(w_candidate, self.p_val)
                self.log_merit = log_merit.full()[0][0]
                self.dlog_merit_x = dlog_merit_x.full()

                ## LINE SEARCH + fraction to boundary
                tau_j = max(solver_opts.tau_min, 1-self.tau_val) # idea: put min(..., 0.999) around?

                # compute new nlp residual after mu step
                # NOTE: not really necessary, maybe make optional
                # kkt_val = self.kkt_eq_fun(w_candidate, self.p_val)
                # nlp_res = ca.norm_inf(kkt_val).full()[0][0]

                # fraction to boundary G, s > 0
                G_val = self.G_fun(w_current).full().flatten()
                G_delta_val = self.G_fun(step).full().flatten()
                alpha_max = self.get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)

                if any(G_val < 0):
                    print("G_val < 0 should never happen")
                    breakpoint()

                # compute minimal step size
                alpha_k_min = solver_opts.gamma_alpha * solver_opts.gamma_theta
                delta_log_merit = (self.dlog_merit_x @ step[:self.nw])[0]
                if delta_log_merit < 0 and self.theta_current <= theta_min:
                    alpha_k_min = min(alpha_k_min,
                            solver_opts.gamma_alpha * solver_opts.gamma_phi * self.theta_current / (-delta_log_merit),
                            solver_opts.gamma_alpha * solver_opts.delta * self.theta_current ** solver_opts.s_theta / (-delta_log_merit) ** solver_opts.s_phi
                            )
                elif delta_log_merit < 0 and self.theta_current > theta_min:
                    alpha_k_min = min(alpha_k_min,
                                solver_opts.gamma_alpha * solver_opts.gamma_phi * self.theta_current / (-delta_log_merit)
                                )
                # line search:
                alpha = alpha_max
                soc_iter = 0
                alpha_max_no_soc = 0. # None
                ls_iter = 0

                # IPOPT: Handling Very Small Search Directions
                eps_mach = 1e-16
                smallness_w = np.max((np.abs(step[:self.nw])) / 1+np.abs(w_current[:self.nw]))
                smallness_s = np.max((np.abs(self.get_slack(step))) / 1+np.abs( self.get_slack(w_current)))
                smallness = max(smallness_w, smallness_s)
                if smallness < 10 * eps_mach:
                    # TODO: apply alpha_k_max
                    breakpoint()

                while True:
                    w_candidate[:self.n_all_but_mu] = w_current[:self.n_all_but_mu] + alpha * step[:self.n_all_but_mu]
                    # t0_ca = time.time()
                    theta_step = self.violation_norm_fun(w_candidate, self.p_val).full()[0][0]
                    step_log_merit = self.log_merit_fun(w_candidate, self.p_val).full()[0][0]
                    dir_der_log_merit = self.dlog_merit_x @ step[:self.nw]

                    # check acceptibility to filter
                    filter_acceptable = self.filter.is_acceptable((theta_step, step_log_merit))
                    if filter_acceptable:
                        # A-5.4 check sufficient decrease
                        if self.theta_current <= theta_min and self.check_Waechter19(step, alpha, dir_der_log_merit):
                            if self.check_Waechter20(step_log_merit, step, alpha, dir_der_log_merit):
                                break # accept
                        elif filter_acceptable:
                            break

                    elif soc_iter > 0:
                        # if not accaptable and already in SOC -> reduce step (A 5.7)
                        alpha *= solver_opts.rho
                        if alpha < alpha_k_min:
                            self.alpha_min_counter += 1
                            print(f"SOC {soc_iter} at it {ii} {newton_iter} alpha {alpha:.2e} alpha_max {alpha_max:.2e} alpha_max_no_soc {alpha_max_no_soc:.2e} theta {self.theta_current:.4e} Dtheta {theta_step:.4e} Del_theta_no_soc {theta_step_no_SOC:.4e} delta_phi {step_log_merit:.2e} phi {self.log_merit:.2e}")
                            self.print_iterates([w_current, step_no_soc, step])
                            breakpoint()
                            self.restoration_phase(w_current)
                        continue

                    # A-5.5 Initialize SOC
                    if soc_iter == 0:
                        if (ls_iter > 0):
                            do_soc = False
                        elif (theta_step < self.theta_current):
                            do_soc = False
                        else:
                            do_soc = True
                            theta_old_soc = self.theta_current
                            theta_step_no_SOC = theta_step
                    else:
                        # A-5.9. Next SOC
                        if soc_iter == solver_opts.max_soc or theta_step > solver_opts.kappa_soc * theta_old_soc:
                            # abort SOC: continue with current one.
                            do_soc = False
                        else:
                            do_soc = True
                    # NOTE: to get multiple SOC: we need that theta is decreasing within SOC, but not "too much" in the kappa_soc sense.

                    # A-5.6 / A-5.9 Perform SOC iter
                    if do_soc:
                        step_no_soc = step.copy()
                        alpha_max_no_soc = alpha_max
                        step = self.get_second_order_correction(w_current, w_candidate, alpha, soc_iter)
                        soc_iter += 1
                        G_delta_val = self.G_fun(step).full().flatten()
                        alpha = self.get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)
                        alpha_max = alpha
                    else:
                        # A-5.10 reduce step
                        alpha *= solver_opts.rho
                        if alpha < alpha_k_min:
                            breakpoint()
                            self.alpha_min_counter += 1
                            print(f"minimum step at it {ii} {newton_iter} alpha {alpha:.2e} alpha_max {alpha_max:.2e} alpha_max_no_soc {alpha_max_no_soc:.2e} theta {self.theta_current:.4e} Dtheta {theta_step:.4e} delta_phi {step_log_merit:.2e} phi {self.log_merit:.2e}")
                            self.print_iterates([w_current, step_no_soc, step])
                            self.restoration_phase(w_current)
                            # self.print_iterates([w_current, step])
                            break
                    #
                    ls_iter += 1

                if soc_iter:
                    print(f"SOC {soc_iter} at it {ii} {newton_iter} alpha {alpha:.2e} alpha_max {alpha_max:.2e} alpha_max_no_soc {alpha_max_no_soc:.2e} theta {self.theta_current:.4e} Dtheta {theta_step:.4e} Del_theta_no_soc {theta_step_no_SOC:.4e} delta_phi {step_log_merit:.2e} phi {self.log_merit:.2e}")
                    if theta_step_no_SOC < theta_step:
                        print("SOC Warning: violation did not decrease")

                # compute alpha_mu_k
                alpha_mu = self.get_fraction_to_boundary(tau_j, w_current[-n_mu:], step[-n_mu:], offset=None)
                w_candidate[-n_mu:] = w_current[-n_mu:] + alpha_mu * step[-n_mu:]

                # do step
                w_current = w_candidate.copy()
                # Waechter2006 eq (16): "Primal-dual barrier term Hessian should not deviate arbitrarily much from primal Hessian"
                G_val = self.G_fun(w_current).full().flatten()
                for i in range(n_mu):
                    new = max(min(w_current[-n_mu+i], solver_opts.kappaSigma * self.tau_val / G_val[i]),
                                                      self.tau_val / (solver_opts.kappaSigma * G_val[i]))
                    if new != w_current[-n_mu+i]:
                        print(f"mu[{i}] = {w_current[-n_mu+i]:e} -> {new:e}")
                    w_current[-n_mu+i] = new
                t_ls += time.time() - t0_ls

                # augment filter if Waechter_19 or Waechter_20 do NOT hold.
                # TODO: avoid reevalution of functions and conditions
                self.theta_current = self.violation_norm_fun(w_current, self.p_val).full()[0][0]
                log_merit_current = self.log_merit_fun(w_candidate, self.p_val).full()[0][0]
                if (not self.check_Waechter19(step, alpha, dir_der_log_merit)) or \
                   (not self.check_Waechter20(step_log_merit, step, alpha, dir_der_log_merit)):
                    self.filter.add((self.theta_current, log_merit_current))

                if opts.print_level > 1:
                    min_mu = np.min(self.get_mu(w_current))
                    max_mu = np.max(self.get_mu(w_current))
                    max_slack_comp_viol = np.max(np.abs(kkt_val[self.nw+self.n_H:self.nw+self.n_H+self.n_comp]))
                    max_H_viol = np.max(np.abs(kkt_val[self.nw:self.nw+self.n_H]))
                    max_comp_viol = np.max(np.abs(kkt_val[:-self.n_mu]))
                    # min_lam_comp = np.min(self.get_lambda_comp(w_current))
                    print(f"{alpha:.3f}\t{alpha_mu:.3f}\t\t{alpha_max:.3f}\t\t{step_norm:.2e}\t{nlp_res:.2e}\t{min_mu:.2e}\t{np.min(G_val):.2e}\t{max_slack_comp_viol:.2e}\t{max_H_viol:.2e}\t{max_comp_viol:.2e}")

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
                print(f"{sigma_k:.2e}\t{newton_iter}\t{nlp_res:.2e}\t{self.alpha_min_counter}\t\t {min_mu:.2e}\t{np.min(G_val):.2e}")

            # complementarity_residual = prob.comp_res(w_current[:self.nw], self.p_val).full()[0][0]
            # complementarity_stats[ii] = complementarity_residual
            # if complementarity_residual < opts.comp_tol:
            #     break
            if ii < opts.max_iter_homotopy - 1:
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

        lam_max = np.max(np.abs(w_current[self.nw:self.nw+self.n_H]))
        mu_max = np.max(self.get_mu(w_current))
        s_d = max(s_max, (lam_max + mu_max) / (self.nw+self.n_H+self.n_comp)) / s_max
        s_c = max(s_max, mu_max/self.nw) / s_max

        scaled_stat = ca.norm_inf(kkt_val[:self.nw]).full()[0][0] / s_d
        scaled_comp = ca.norm_inf(kkt_val[-self.nG-self.n_comp:]).full()[0][0] / s_c
        eq_res = ca.norm_inf(kkt_val[self.nw:self.nw+self.n_H]).full()[0][0]

        if max(scaled_comp, eq_res) < solver_opts.kappa_res_sigma * sigma_k:
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

        return results

    def print_iterates_w(self, iterate_list: list):
        for ii in range(self.nw):
            line = f"{ii}\t{self.w_pd[ii].name():17}"
            for it in iterate_list:
                line += f'\t{it[ii]:.4e}'
            print(line)

    def restoration_phase(self, w_current: np.ndarray):
        print("in restoration phase")
        solver_opts = self.solver_opts

        ## initialize

        x_resto_current = np.concatenate((w_current[:self.nw], self.get_slack(w_current)))
        x_resto_ref = x_resto_current.copy()
        nx_resto = len(x_resto_ref)
        rho_resto = 1e3

        # setup parameter! keep sigma fixed
        tau_val = max(self.tau_val, self.theta_current)
        zeta_val = np.sqrt(tau_val) # TODO: update / eliminate?
        D_resto = np.array([max(1, 1/np.abs(x_resto_ref[i])) for i in range(nx_resto)])
        resto_param_val = np.concatenate((self.p_val[:-1].copy(), np.array([tau_val, zeta_val, rho_resto]), x_resto_ref, D_resto))

        # initialize p, n
        # eval constraint_violation
        c_val = self.resto_c_fun(x_resto_current, resto_param_val).full().flatten()

        n_val, p_val = self.restoration_get_np_values(c_val, tau_val)

        # duals for n, p
        z_n_val = tau_val * (1/n_val)
        z_p_val = tau_val * (1/p_val)
        lambda0 = np.zeros(self.n_H+self.n_comp) # lambda_H; lambda_slacked_comp (= mu_s)
        mu_outside = self.get_mu(w_current)
        # TODO: mu_s is used here twice, multiplier for G_s and slacked_complementarity; initialize as z or lambda?
        mu_G = mu_outside[:self.nG]
        mu_s = mu_outside[self.nG:]
        # lambda_0 from outside would be: w_current[self.nw:self.nw+self.n_H]

        z_val_G = np.array([min(rho_resto, mu_G[i]) for i in range(self.nG)]) # z_k from outside # multipliers of G
        z_val_s = np.array([min(rho_resto, mu_s[i]) for i in range(self.n_comp)])

        resto_duals = np.concatenate((lambda0, z_val_G, z_val_s, z_n_val, z_p_val))

        w_candidate = w_current.copy()
        # TODO: be careful with w_current!

        # initialize filter
        theta_max = 1e4 * max(1, self.theta_current)
        self.resto_filter = NosnocFilter(theta_max=theta_max, gamma_theta=solver_opts.gamma_theta)

        ## apply "normal" IP algorithm
        for newton_iter in range(solver_opts.max_newton_iter):

            ## check termination
            # update w_candidate (x, slack)
            w_candidate[:self.nw] = x_resto_current[:self.nw]
            w_candidate[self.nw+self.n_H:self.nw+self.n_H+self.n_comp] = x_resto_current[self.nw:]
            # i) is acceptible to (outside) filter
            theta_step = self.violation_norm_fun(w_candidate, self.p_val).full()[0][0]
            step_log_merit = self.log_merit_fun(w_candidate, self.p_val).full()[0][0]
            if self.filter.is_acceptable((theta_step, step_log_merit)):
                # ii) if sufficient decrease in violation
                print("filter accepts, lets check sufficient decrease in violation")
                if theta_step < self.theta_current * solver_opts.kappa_resto:
                    print("restoration success")
                    breakpoint()

            # setup linear system
            mat, rhs, G_val = self.ls_resto_fun(x_resto_current, np.concatenate((n_val, p_val)), resto_duals, resto_param_val)
            G_val = G_val.full().flatten()

            # solve
            lu_fact_resto = scipy.sparse.linalg.factorized(mat.sparse())
            sol = lu_fact_resto(rhs.full().flatten())
            dx = sol[:nx_resto]
            d_lambda = sol[nx_resto:]

            # expand
            # dp = (tau + diag()
            dp = (tau_val + p_val * (lambda0 + d_lambda) - rho_resto * p_val) / z_p_val
            dn = (tau_val + n_val * (lambda0 + d_lambda) - rho_resto * n_val) / z_n_val
            dzp = tau_val / p_val - z_p_val - z_p_val / p_val * dp
            dzn = tau_val / n_val - z_n_val - z_n_val / n_val * dn
            z_by_G = np.concatenate((z_val_G, z_val_s)) / G_val
            dz = tau_val / G_val - G_val - (z_by_G * (self.dG_dx @ dx))

            # line search
            ## LINE SEARCH + fraction to boundary
            tau_j = max(solver_opts.tau_min, 1-tau_val)
            w_candidate = w_current.copy()
            w_candidate[:self.nw] += dx[:self.nw]
            # update slack
            w_candidate[self.nw+self.n_H: self.nw+self.n_H+self.n_comp] += dx[self.nw:]

            # fraction to boundary G, s, n, p > 0
            G_val = self.G_fun(w_current).full().flatten()
            G_delta_val = self.G_fun(w_candidate).full().flatten()
            alpha_max = self.get_fraction_to_boundary(tau_j, G_val, G_delta_val, offset=self.G_offset)
            alpha_max_np = self.get_fraction_to_boundary(tau_j, np.concatenate((n_val, p_val)), np.concatenate((dn, dp)))
            print(f"{alpha_max=}, {alpha_max_np=}")

            self.print_iterates_w([w_current, dx])

            if any(G_val < 0):
                print("G_val < 0 should never happen")

            self.resto_print_np(c_val, p_val, dp, n_val, dn)
            breakpoint()

            if alpha_max_np < 0.1:
                print()

            # line search:
            alpha = alpha_max
            soc_iter = 0
            alpha_max_no_soc = 0. # None
            ls_iter = 0

            # while True:

            # do step

            pass

    def resto_print_np(self, c_val, p_val, dp, n_val, dn):
        print("\nconstraint c, n, p values")
        print("c\t\tp\t\tdp\t\tn\t\tdn")
        for i in range(len(c_val)):
            print(f"{c_val[i]:.2e}\t{p_val[i]:.2e}\t{dp[i]:.2e}\t{n_val[i]:.2e}\t{dn[i]:.2e}")

    def restoration_get_np_values(self, c_val, tau: float):
        n_c = len(c_val)
        rho_resto = self.solver_opts.rho_resto

        n_val = np.array((n_c,))
        p_val = np.array((n_c,))
        n_val = np.array([(tau - rho_resto * c_val[ii]) / (2*rho_resto) + \
                np.sqrt( ((tau - rho_resto * c_val[ii]) / (2*rho_resto))**2 + (tau * c_val[ii])/(2*rho_resto) ) for ii in range(n_c)])
        p_val = c_val + n_val

        return n_val, p_val

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
