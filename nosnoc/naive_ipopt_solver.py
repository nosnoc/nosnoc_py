from typing import Optional

import casadi as ca
import numpy as np

from .solver import NosnocSolverBase, get_results_from_primal_vector
from .model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts

from nosnoc.nosnoc_types import InitializationStrategy, PssMode, ConstraintHandling, Status
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.utils import check_ipopt_success


class NaiveIpoptSolver(NosnocSolverBase):
    """
    An alternative to the NosnocSolver that uses multiple Ipopt with default options.
    """

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):

        super().__init__(opts, model, ocp)

        # dont use opts.opts_casadi_nlp
        opts_casadi_nlp = dict()
        opts_casadi_nlp['print_time'] = 0
        opts_casadi_nlp["record_time"] = True
        opts_casadi_nlp['verbose'] = False
        opts_casadi_nlp['ipopt'] = dict()
        opts_casadi_nlp['ipopt']['print_level'] = 0

        # create NLP Solver
        try:
            casadi_nlp = {
                'f': self.problem.cost,
                'x': self.problem.w,
                'g': self.problem.g,
                'p': self.problem.p
            }
            self.solver = ca.nlpsol(model.name, 'ipopt', casadi_nlp, opts_casadi_nlp)
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

        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = min(sigma_k ** 1.5, sigma_k)
            # tau_val = sigma_k**1.5*1e3
            self.setup_p_val(sigma_k, tau_val)

            # solve NLP
            if ii > 0:
                sol = self.solver(x0=w0,
                                lbg=prob.lbg,
                                ubg=prob.ubg,
                                lbx=prob.lbw,
                                ubx=prob.ubw,
                                p=self.p_val,
                                lam_g0 = sol['lam_g'],
                                lam_x0 = sol['lam_x'])
            else:
                sol = self.solver(x0=w0,
                    lbg=prob.lbg,
                    ubg=prob.ubg,
                    lbx=prob.lbw,
                    ubx=prob.ubw,
                    p=self.p_val,)
            # statistics
            solver_stats = self.solver.stats()
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
            w_opt, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish - 1], status = \
                                            self.polish_solution(self.solver, w_opt)

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
