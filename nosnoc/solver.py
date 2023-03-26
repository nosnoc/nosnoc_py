
from abc import ABC, abstractmethod
from typing import Optional

import casadi as ca
import numpy as np
import time

from nosnoc.model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import InitializationStrategy, PssMode, HomotopyUpdateRule, ConstraintHandling, Status
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.rk_utils import rk4_on_timegrid
from nosnoc.utils import flatten_layer, flatten, get_cont_algebraic_indices, flatten_outer_layers, check_ipopt_success


class NosnocSolverBase(ABC):

    @abstractmethod
    def __init__(self,
                 opts: NosnocOpts,
                 model: NosnocModel,
                 ocp: Optional[NosnocOcp] = None) -> None:
        # preprocess inputs
        opts.preprocess()
        model.preprocess_model(opts)

        if opts.initialization_strategy == InitializationStrategy.RK4_SMOOTHENED:
            model.add_smooth_step_representation(smoothing_parameter=opts.smoothing_parameter)

        # store references
        self.model = model
        self.ocp = ocp
        self.opts = opts

        # create problem
        problem = NosnocProblem(opts, model, ocp)
        self.problem: NosnocProblem = problem
        return

    def set(self, field: str, value: np.ndarray) -> None:
        """
        Set values.

        :param field: in ["x0", "x", "u", "p_global", "p_time_var", "w"]
        :param value: np.ndarray: numerical value of appropriate size
        """
        prob = self.problem
        dims = prob.model.dims
        if field == 'x0':
            prob.model.x0 = value
        elif field == 'x': # TODO: check other dimensions for useful error message
            if value.shape[0] == self.opts.N_stages:
                # Shape is equal to the number of control stages
                for i, sub_idx in enumerate(prob.ind_x):
                    for ssub_idx in sub_idx:
                        for sssub_idx in ssub_idx:
                            prob.w0[sssub_idx] = value[i, :]
            elif value.shape[0] == sum(self.opts.Nfe_list):
                # Shape is equal to the number of finite elements
                i = 0
                for sub_idx in prob.ind_x:
                    for ssub_idx in sub_idx:
                        for sssub_idx in ssub_idx:
                            prob.w0[sssub_idx] = value[i, :]
                        i += 1
            else:
                raise ValueError("value should have shape matching N_stages "
                                 f"({self.opts.N_stages}) or sum(Nfe_list) "
                                 f"({sum(self.opts.Nfe_list)}), shape: {value.shape[0]}")

            if self.opts.initialization_strategy is not InitializationStrategy.EXTERNAL:
                raise Warning(
                    'initialization of x might be overwritten due to InitializationStrategy != EXTERNAL.'
                )
        elif field == 'u':
            prob.w0[prob.ind_u] = value
        elif field == 'p_global':
            for i in range(self.opts.N_stages):
                self.model.p_val_ctrl_stages[i, dims.n_p_time_var:] = value
        elif field == 'p_time_var':
            for i in range(self.opts.N_stages):
                self.model.p_val_ctrl_stages[i, :dims.n_p_time_var] = value[i, :]
        elif field == 'w':
            prob.w0 = value
            if self.opts.initialization_strategy is not InitializationStrategy.EXTERNAL:
                raise Warning(
                    'full initialization w might be overwritten due to InitializationStrategy != EXTERNAL.'
                )
        else:
            raise NotImplementedError()

    def print_problem(self):
        self.problem.print()

    def initialize(self):
        opts = self.opts
        prob = self.problem
        x0 = prob.model.x0

        if opts.initialization_strategy in [
                InitializationStrategy.ALL_XCURRENT_W0_START,
                InitializationStrategy.ALL_XCURRENT_WOPT_PREV
        ]:
            for ind in prob.ind_x:
                prob.w0[ind] = x0
        elif opts.initialization_strategy == InitializationStrategy.EXTERNAL:
            pass
        # This is experimental
        elif opts.initialization_strategy == InitializationStrategy.RK4_SMOOTHENED:
            # print(f"updating w0 with RK4 smoothened")
            # NOTE: assume N_stages = 1 and STEWART
            dt_fe = opts.terminal_time / (opts.N_stages * opts.N_finite_elements)
            irk_time_grid = np.array(
                [opts.irk_time_points[0]] +
                [opts.irk_time_points[k] - opts.irk_time_points[k - 1] for k in range(1, opts.n_s)])
            rk4_t_grid = dt_fe * irk_time_grid

            x_rk4_current = x0
            db_updated_indices = list()
            for i in range(opts.N_finite_elements):
                Xrk4 = rk4_on_timegrid(self.model.f_x_smooth_fun,
                                       x0=x_rk4_current,
                                       t_grid=rk4_t_grid)
                x_rk4_current = Xrk4[-1]
                # print(f"{Xrk4=}")
                for k in range(opts.n_s):
                    x_ki = Xrk4[k + 1]
                    prob.w0[prob.ind_x[0][i][k]] = x_ki
                    # NOTE: we don't use lambda_smooth_fun, since it gives negative lambdas
                    # -> infeasible. Possibly another smooth min fun could be used.
                    # However, this would be inconsistent with mu.
                    p_0 = self.model.p_val_ctrl_stages[0]
                    lam_ki = self.model.lambda00_fun(x_ki, p_0)
                    mu_ki = self.model.mu_smooth_fun(x_ki, p_0)
                    theta_ki = self.model.theta_smooth_fun(x_ki, p_0)
                    # print(f"{x_ki=}")
                    # print(f"theta_ki = {list(theta_ki.full())}")
                    # print(f"mu_ki = {list(mu_ki.full())}")
                    # print(f"lam_ki = {list(lam_ki.full())}\n")
                    for s in range(self.model.dims.n_sys):
                        ind_theta_s = range(sum(self.model.dims.n_f_sys[:s]),
                                            sum(self.model.dims.n_f_sys[:s + 1]))
                        prob.w0[prob.ind_theta[0][i][k][s]] = theta_ki[ind_theta_s].full().flatten()
                        prob.w0[prob.ind_lam[0][i][k][s]] = lam_ki[ind_theta_s].full().flatten()
                        prob.w0[prob.ind_mu[0][i][k][s]] = mu_ki[s].full().flatten()
                        # TODO: ind_v
                    db_updated_indices += prob.ind_theta[0][i][k][s] + prob.ind_lam[0][i][k][
                        s] + prob.ind_mu[0][i][k][s] + prob.ind_x[0][i][k] + prob.ind_h
                if opts.irk_time_points[-1] != 1.0:
                    raise NotImplementedError
                else:
                    # Xk_end
                    prob.w0[prob.ind_x[0][i][-1]] = x_ki
                    db_updated_indices += prob.ind_x[0][i][-1]

            # print("w0 after RK4 init:")
            # print(prob.w0)
            missing_indices = sorted(set(range(len(prob.w0))) - set(db_updated_indices))
            # print(f"{missing_indices=}")

    def _print_iter_stats(self, sigma_k, complementarity_residual, nlp_res, cost_val, cpu_time_nlp,
                          nlp_iter, status):
        print(f'{sigma_k:.1e} \t {complementarity_residual:.2e} \t {nlp_res:.2e}' +
              f'\t {cost_val:.2e} \t {cpu_time_nlp:3f} \t {nlp_iter} \t {status}')


class NosnocSolver(NosnocSolverBase):
    """
    Main solver class which solves the nonsmooth problem by applying a homotopy
    and solving the NLP subproblems using IPOPT.

    The nonsmooth problem is formulated internally based on the given options,
    dynamic model, and (optionally) the ocp data.
    """

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        """Constructor.
        """
        super().__init__(opts, model, ocp)

        # create NLP Solver
        try:
            casadi_nlp = {
                'f': self.problem.cost,
                'x': self.problem.w,
                'g': self.problem.g,
                'p': self.problem.p
            }
            self.solver = ca.nlpsol(model.name, 'ipopt', casadi_nlp, opts.opts_casadi_nlp)
        except Exception as err:
            self.print_problem()
            print(f"{opts=}")
            print("\nerror creating solver for problem above.")
            raise err

    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.

        :return: Returns a dictionary containing ... TODO document all fields
        """
        self.initialize()
        opts = self.opts
        prob = self.problem
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

        # lambda00 initialization
        x0 = prob.model.x0
        lambda00 = prob.model.get_lambda00(opts)

        if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:
            lbw = prob.lbw.copy()
            ubw = prob.ubw.copy()

            # lambda00 != 0.0 -> corresponding thetas on first fe are zero
            I_active_lam = np.where(lambda00 > 1e1*opts.comp_tol)[0].tolist()
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
            # I_lam_zero = set(range(len(lambda00))).difference( I_active_lam )
            # n_lam = sum(prob.model.dims.n_f_sys)
            # if len(I_active_lam) == n_lam - 1:
            #     for i in range(opts.n_s):
            #         tmp = flatten(ind_theta_fe1[i])
            #         w_one_indices += [tmp[i] for i in I_lam_zero]
            if opts.print_level > 1:
                print(f"fixing {prob.w[w_one_indices]} = 1. and {prob.w[w_zero_indices]} = 0.")
                print(f"Since lambda00 = {lambda00}")
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
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(), np.array([sigma_k,
                                                                   tau_val]), lambda00, x0))

            # solve NLP
            t = time.process_time()
            sol = self.solver(x0=w0,
                              lbg=prob.lbg,
                              ubg=prob.ubg,
                              lbx=lbw,
                              ubx=ubw,
                              p=p_val)
            cpu_time_nlp[ii] = time.process_time() - t

            # print and process solution
            solver_stats = self.solver.stats()
            status = solver_stats['return_status']
            nlp_iter[ii] = solver_stats['iter_count']
            nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_opt = sol['x'].full().flatten()
            w0 = w_opt
            w_all.append(w_opt)

            complementarity_residual = prob.comp_res(w_opt, p_val).full()[0][0]
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
            if opts.homotopy_update_rule == HomotopyUpdateRule.LINEAR:
                sigma_k = opts.homotopy_update_slope * sigma_k
            elif opts.homotopy_update_rule == HomotopyUpdateRule.SUPERLINEAR:
                sigma_k = max(
                    opts.sigma_N,
                    min(opts.homotopy_update_slope * sigma_k,
                        sigma_k**opts.homotopy_update_exponent))

        if opts.do_polishing_step:
            w_opt, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish - 1], status = \
                                            self.polish_solution(w_opt, lambda00, x0)

        # collect results
        results = get_results_from_primal_vector(prob, w_opt)

        # print constraint violation
        if opts.print_level > 1 and opts.constraint_handling == ConstraintHandling.LEAST_SQUARES:
            threshold = np.max([np.sqrt(cost_val) / 100, opts.comp_tol * 1e2, 1e-5])
            g_val = prob.g_fun(w_opt, p_val).full().flatten()
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
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(), np.array([sigma_k,
                                                                   tau_val]), lambda00, x0))

            # solve NLP
            t = time.time()
            sol = self.solver(x0=w_guess, lbg=prob.lbg, ubg=prob.ubg, lbx=lbw, ubx=ubw, p=p_val)
            cpu_time_nlp = time.time() - t

            # print and process solution
            solver_stats = self.solver.stats()
            status = solver_stats['return_status']
            nlp_iter = solver_stats['iter_count']
            nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_opt = sol['x'].full().flatten()

            complementarity_residual = prob.comp_res(w_opt, p_val).full()[0][0]
            if opts.print_level:
                self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
                                       cpu_time_nlp, nlp_iter, status)
            if status not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Warning: IPOPT exited with status {status}")

        return w_opt, cpu_time_nlp, nlp_iter, status


def get_results_from_primal_vector(prob: NosnocProblem, w_opt: np.ndarray) -> dict:
    opts = prob.opts

    results = dict()
    results["x_out"] = w_opt[prob.ind_x[-1][-1][-1]]
    # TODO: improve naming here?
    results["x_list"] = [w_opt[ind] for ind in flatten_layer(prob.ind_x_cont)]

    x0 = prob.model.x0
    ind_x_all = flatten_outer_layers(prob.ind_x, 2)
    results["x_all_list"] = [x0] + [w_opt[ind] for ind in ind_x_all]
    results["u_list"] = [w_opt[ind] for ind in prob.ind_u]

    results["theta_list"] = [w_opt[ind] for ind in get_cont_algebraic_indices(prob.ind_theta)]
    results["lambda_list"] = [w_opt[ind] for ind in get_cont_algebraic_indices(prob.ind_lam)]
    # results["mu_list"] = [w_opt[ind] for ind in ind_mu_all]
    # if opts.pss_mode == PssMode.STEP:
    results["alpha_list"] = [
        w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_alpha)
    ]
    results["lambda_n_list"] = [
        w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_lambda_n)
    ]
    results["lambda_p_list"] = [
        w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_lambda_p)
    ]

    if opts.use_fesd:
        time_steps = w_opt[prob.ind_h]
    else:
        t_stages = opts.terminal_time / opts.N_stages
        for Nfe in opts.Nfe_list:
            time_steps = Nfe * [t_stages / Nfe]
    results["time_steps"] = time_steps

    # results relevant for OCP:
    results["x_traj"] = [x0] + results["x_list"]
    results["u_traj"] = results["u_list"]  # duplicate name
    t_grid = np.concatenate((np.array([0.0]), np.cumsum(time_steps)))
    results["t_grid"] = t_grid
    u_grid = [0] + np.cumsum(opts.Nfe_list).tolist()
    results["t_grid_u"] = [t_grid[i] for i in u_grid]

    results["v_global"] = w_opt[prob.ind_v_global]

    # NOTE: this doesn't handle sliding modes well. But seems nontrivial.
    # compute based on changes in alpha or theta
    switch_indices = []
    if opts.pss_mode == PssMode.STEP:
        alpha_prev = results["alpha_list"][0]
        for i, alpha in enumerate(results["alpha_list"][1:]):
            if any(np.abs(alpha - alpha_prev) > 0.1):
                switch_indices.append(i)
            alpha_prev = alpha
    else:
        theta_prev = results["theta_list"][0]
        for i, theta in enumerate(results["theta_list"][1:]):
            if any(np.abs(theta.flatten() - theta_prev.flatten()) > 0.1):
                switch_indices.append(i)
            theta_prev = theta

    results["switch_times"] = np.array([time_steps[i] for i in switch_indices])

    return results
