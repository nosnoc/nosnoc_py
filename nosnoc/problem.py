from typing import Optional, List
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import casadi as ca

from nosnoc.model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, ConstraintHandling
from nosnoc.ocp import NosnocOcp
from nosnoc.utils import casadi_length, casadi_vertcat_list, casadi_sum_list, flatten, increment_indices, create_empty_list_matrix


class NosnocFormulationObject(ABC):

    @abstractmethod
    def __init__(self):
        # optimization variables with initial guess, bounds
        self.w: ca.SX = ca.SX([])
        self.w0: np.array = np.array([])
        self.lbw: np.array = np.array([])
        self.ubw: np.array = np.array([])

        # constraints and bounds
        self.g: ca.SX = ca.SX([])
        self.lbg: np.array = np.array([])
        self.ubg: np.array = np.array([])

        # cost
        self.cost: ca.SX = ca.SX.zeros(1)

        # index lists
        self.ind_x: list
        self.ind_lam: list
        self.ind_lambda_n: list
        self.ind_lambda_p: list

    def __repr__(self):
        return repr(self.__dict__)

    def add_variable(self,
                     symbolic: ca.SX,
                     index: list,
                     lb: np.array,
                     ub: np.array,
                     initial: np.array,
                     stage: Optional[int] = None,
                     sys: Optional[int] = None):
        n = casadi_length(symbolic)
        nw = casadi_length(self.w)

        if len(lb) != n or len(ub) != n or len(initial) != n:
            raise Exception(
                f'add_variable, inconsistent dimension: {symbolic=}, {lb=}, {ub=}, {initial=}')

        self.w = ca.vertcat(self.w, symbolic)
        self.lbw = np.concatenate((self.lbw, lb))
        self.ubw = np.concatenate((self.ubw, ub))
        self.w0 = np.concatenate((self.w0, initial))

        new_indices = list(range(nw, nw + n))
        if stage is None:
            index.append(new_indices)
        else:
            if sys is not None:
                index[stage][sys] = new_indices
            else:
                index[stage] = new_indices
        return

    def add_constraint(self, symbolic: ca.SX, lb=None, ub=None, index: Optional[list] = None):
        n = casadi_length(symbolic)
        if n == 0:
            return
        if lb is None:
            lb = np.zeros((n,))
        if ub is None:
            ub = np.zeros((n,))
        if len(lb) != n or len(ub) != n:
            raise Exception(f'add_constraint, inconsistent dimension: {symbolic=}, {lb=}, {ub=}')

        if index is not None:
            ng = casadi_length(self.g)
            new_indices = list(range(ng, ng + n))
            index.append(new_indices)

        self.g = ca.vertcat(self.g, symbolic)
        self.lbg = np.concatenate((self.lbg, lb))
        self.ubg = np.concatenate((self.ubg, ub))

        return


class FiniteElementBase(NosnocFormulationObject):

    def Lambda(self, stage=slice(None), sys=slice(None)):
        return ca.vertcat(self.w[flatten(self.ind_lam[stage][sys])],
                          self.w[flatten(self.ind_lambda_n[stage][sys])],
                          self.w[flatten(self.ind_lambda_p[stage][sys])])


class FiniteElementZero(FiniteElementBase):

    def __init__(self, opts: NosnocOpts, model: NosnocModel):
        super().__init__()
        dims = model.dims

        self.ind_x = create_empty_list_matrix((1,))
        self.ind_lam = create_empty_list_matrix((1, dims.n_sys))
        self.ind_lambda_n = create_empty_list_matrix((1, dims.n_sys))
        self.ind_lambda_p = create_empty_list_matrix((1, dims.n_sys))

        # NOTE: bounds are actually not used, maybe rewrite without add_vairable
        # X0
        self.add_variable(ca.SX.sym('X0', dims.n_x), self.ind_x, model.x0, model.x0, model.x0, 0)

        # lambda00
        if opts.pss_mode == PssMode.STEWART:
            for ij in range(dims.n_sys):
                self.add_variable(ca.SX.sym(f'lambda00_{ij+1}', dims.n_f_sys[ij]), self.ind_lam,
                                  -np.inf * np.ones(dims.n_f_sys[ij]),
                                  np.inf * np.ones(dims.n_f_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_f_sys[ij]), 0, ij)
        elif opts.pss_mode == PssMode.STEP:
            for ij in range(dims.n_sys):
                self.add_variable(ca.SX.sym(f'lambda00_n_{ij+1}', dims.n_c_sys[ij]),
                                  self.ind_lambda_n, -np.inf * np.ones(dims.n_c_sys[ij]),
                                  np.inf * np.ones(dims.n_c_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_c_sys[ij]), 0, ij)
                self.add_variable(ca.SX.sym(f'lambda00_p_{ij+1}', dims.n_c_sys[ij]),
                                  self.ind_lambda_p, -np.inf * np.ones(dims.n_c_sys[ij]),
                                  np.inf * np.ones(dims.n_c_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_c_sys[ij]), 0, ij)


class FiniteElement(FiniteElementBase):

    def __init__(self,
                 opts: NosnocOpts,
                 model: NosnocModel,
                 ocp: NosnocOcp,
                 ctrl_idx: int,
                 fe_idx: int,
                 prev_fe=None):

        super().__init__()
        n_s = opts.n_s

        # store info
        self.n_rkstages = n_s
        self.ctrl_idx = ctrl_idx
        self.fe_idx = fe_idx
        self.opts = opts
        self.model = model
        self.ocp = ocp
        self.prev_fe: FiniteElementBase = prev_fe
        self.p = model.p_ctrl_stages[ctrl_idx]

        dims = self.model.dims

        # right boundary
        create_right_boundary_point = (opts.use_fesd and not opts.right_boundary_point_explicit and
                                       fe_idx < opts.Nfe_list[ctrl_idx] - 1)
        end_allowance = 1 if create_right_boundary_point else 0

        # Initialze index lists
        if opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
            # only x_end
            self.ind_x = create_empty_list_matrix((1,))
        elif opts.right_boundary_point_explicit:
            self.ind_x = create_empty_list_matrix((n_s,))
        else:
            self.ind_x = create_empty_list_matrix((n_s + 1,))
        self.ind_v: list = create_empty_list_matrix((n_s,))
        self.ind_theta = create_empty_list_matrix((n_s, dims.n_sys))
        self.ind_lam = create_empty_list_matrix((n_s + end_allowance, dims.n_sys))
        self.ind_mu = create_empty_list_matrix((n_s + end_allowance, dims.n_sys))
        self.ind_alpha = create_empty_list_matrix((n_s, dims.n_sys))
        self.ind_lambda_n = create_empty_list_matrix((n_s + end_allowance, dims.n_sys))
        self.ind_lambda_p = create_empty_list_matrix((n_s + end_allowance, dims.n_sys))
        self.ind_z = create_empty_list_matrix((n_s,))
        self.ind_h = []

        self.ind_comp = []

        # create variables
        h = ca.SX.sym(f'h_{ctrl_idx}_{fe_idx}')
        h_ctrl_stage = opts.terminal_time / opts.N_stages
        h0 = np.array([h_ctrl_stage / np.array(opts.Nfe_list[ctrl_idx])])
        ubh = (1 + opts.gamma_h) * h0
        lbh = (1 - opts.gamma_h) * h0
        self.add_step_size_variable(h, lbh, ubh, h0)

        if opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode.SCHOLTES_INEQ]:
            lb_dual = 0.0
        else:
            lb_dual = -np.inf

        # RK stage stuff
        for ii in range(opts.n_s):
            # state derivatives
            if (opts.irk_representation
                    in [IrkRepresentation.DIFFERENTIAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
                self.add_variable(ca.SX.sym(f'V_{ctrl_idx}_{fe_idx}_{ii+1}', dims.n_x), self.ind_v,
                                  -np.inf * np.ones(dims.n_x), np.inf * np.ones(dims.n_x),
                                  np.zeros(dims.n_x), ii)
            # states
            if (opts.irk_representation
                    in [IrkRepresentation.INTEGRAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
                self.add_variable(ca.SX.sym(f'X_{ctrl_idx}_{fe_idx}_{ii+1}', dims.n_x), self.ind_x,
                                  ocp.lbx, ocp.ubx, model.x0, ii)
            # algebraic variables
            if opts.pss_mode == PssMode.STEWART:
                # add thetas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'theta_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_theta, lb_dual * np.ones(dims.n_f_sys[ij]),
                        np.inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_theta * np.ones(dims.n_f_sys[ij]), ii, ij)
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, lb_dual * np.ones(dims.n_f_sys[ij]),
                        np.inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_f_sys[ij]), ii, ij)
                # add mu
                for ij in range(dims.n_sys):
                    self.add_variable(ca.SX.sym(f'mu_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', 1),
                                      self.ind_mu, -np.inf * np.ones(1), np.inf * np.ones(1),
                                      opts.init_mu * np.ones(1), ii, ij)
            elif opts.pss_mode == PssMode.STEP:
                # add alpha
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'alpha_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                                  dims.n_c_sys[ij]), self.ind_alpha,
                        lb_dual * np.ones(dims.n_c_sys[ij]), np.ones(dims.n_c_sys[ij]),
                        opts.init_theta * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_n
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_n_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_c_sys[ij]),
                        self.ind_lambda_n, lb_dual * np.ones(dims.n_c_sys[ij]),
                        np.inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                                  dims.n_c_sys[ij]), self.ind_lambda_p,
                        lb_dual * np.ones(dims.n_c_sys[ij]), np.inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_mu * np.ones(dims.n_c_sys[ij]), ii, ij)
            # user algebraic variables
            self.add_variable(
                ca.SX.sym(f'z_{ctrl_idx}_{fe_idx}_{ii+1}', dims.n_z), self.ind_z,
                model.lbz, model.ubz, model.z0, ii
            )

        # Add right boundary points if needed
        if create_right_boundary_point:
            if opts.pss_mode == PssMode.STEWART:
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_end_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, lb_dual * np.ones(dims.n_f_sys[ij]),
                        np.inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_f_sys[ij]), opts.n_s, ij)
                # add mu
                for ij in range(dims.n_sys):
                    self.add_variable(ca.SX.sym(f'mu_{ctrl_idx}_{fe_idx}_end_{ij+1}', 1),
                                      self.ind_mu, -np.inf * np.ones(1), np.inf * np.ones(1),
                                      opts.init_mu * np.ones(1), opts.n_s, ij)
            elif opts.pss_mode == PssMode.STEP:
                # add lambda_n
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_n_{ctrl_idx}_{fe_idx}_end_{ij+1}', dims.n_c_sys[ij]),
                        self.ind_lambda_n, lb_dual * np.ones(dims.n_c_sys[ij]),
                        np.inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), opts.n_s, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        ca.SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_end_{ij+1}',
                                  dims.n_c_sys[ij]), self.ind_lambda_p,
                        lb_dual * np.ones(dims.n_c_sys[ij]), np.inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_mu * np.ones(dims.n_c_sys[ij]), opts.n_s, ij)

        if (not opts.right_boundary_point_explicit or
                opts.irk_representation == IrkRepresentation.DIFFERENTIAL):
            # add final X variables
            self.add_variable(ca.SX.sym(f'X_end_{ctrl_idx}_{fe_idx+1}', dims.n_x), self.ind_x,
                              ocp.lbx, ocp.ubx, model.x0, -1)

    def add_step_size_variable(self, symbolic: ca.SX, lb: float, ub: float, initial: float):
        self.ind_h = casadi_length(self.w)
        self.w = ca.vertcat(self.w, symbolic)

        self.lbw = np.append(self.lbw, lb)
        self.ubw = np.append(self.ubw, ub)
        self.w0 = np.append(self.w0, initial)
        return

    def rk_stage_z(self, stage) -> ca.SX:
        idx = np.concatenate((flatten(self.ind_theta[stage]), flatten(self.ind_lam[stage]),
                              flatten(self.ind_mu[stage]), flatten(self.ind_alpha[stage]),
                              flatten(self.ind_lambda_n[stage]), flatten(self.ind_lambda_p[stage]),
                              self.ind_z[stage]))
        return self.w[idx]

    def Theta(self, stage=slice(None), sys=slice(None)) -> ca.SX:
        return ca.vertcat(
            self.w[flatten(self.ind_theta[stage][sys])],
            self.w[flatten(self.ind_alpha[stage][sys])],
            np.ones(len(flatten(self.ind_alpha[stage][sys]))) -
            self.w[flatten(self.ind_alpha[stage][sys])])

    def get_Theta_list(self) -> list:
        return [self.Theta(stage=ii) for ii in range(len(self.ind_theta))]

    def sum_Theta(self) -> ca.SX:
        return casadi_sum_list(self.get_Theta_list())

    def get_Lambdas_incl_last_prev_fe(self, sys=slice(None)):
        Lambdas = [self.Lambda(stage=ii, sys=sys) for ii in range(len(self.ind_lam))]
        Lambdas += [self.prev_fe.Lambda(stage=-1, sys=sys)]
        return Lambdas

    def sum_Lambda(self, sys=slice(None)):
        """NOTE: includes the prev fes last stage lambda"""
        Lambdas = self.get_Lambdas_incl_last_prev_fe(sys)
        return casadi_sum_list(Lambdas)

    def h(self) -> List[ca.SX]:
        return self.w[self.ind_h]

    def X_fe(self) -> ca.SX:
        opts = self.opts
        if opts.irk_representation == IrkRepresentation.INTEGRAL:
            X_fe = [self.w[ind] for ind in self.ind_x]
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
            X_fe = []
            for j in range(opts.n_s):
                x_temp = self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r in range(opts.n_s):
                    x_temp += self.h() * opts.A_irk[j, r] * self.w[self.ind_v[r]]
                X_fe.append(x_temp)
            X_fe.append(self.w[self.ind_x[-1]])
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL_LIFT_X:
            X_fe = [self.w[ind] for ind in self.ind_x]

        return X_fe

    def forward_simulation(self, ocp: NosnocOcp, Uk: ca.SX) -> None:
        opts = self.opts
        model = self.model

        # setup X_fe: list of x values on fe, initialize X_end
        # TODO: clean up
        X_fe = self.X_fe()
        if opts.irk_representation == IrkRepresentation.INTEGRAL:
            Xk_end = opts.D_irk[0] * self.prev_fe.w[self.prev_fe.ind_x[-1]]
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
            Xk_end = self.prev_fe.w[self.prev_fe.ind_x[-1]]
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL_LIFT_X:
            Xk_end = self.prev_fe.w[self.prev_fe.ind_x[-1]]
            for j in range(opts.n_s):
                x_temp = self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r in range(opts.n_s):
                    x_temp += self.h() * opts.A_irk[j, r] * self.w[self.ind_v[r]]
                # lifting constraints
                self.add_constraint(self.w[self.ind_x[j]] - x_temp)

        for j in range(opts.n_s):
            # Dynamics excluding complementarities
            fj = model.f_x_fun(X_fe[j], self.rk_stage_z(j), Uk, self.p, model.v_global)
            qj = ocp.f_q_fun(X_fe[j], Uk, self.p, model.v_global)
            gqj = ocp.g_path_fun(X_fe[j], Uk, self.p, model.v_global)
            self.add_constraint(gqj, ocp.lbg, ocp.ubg)

            # path constraint
            gj = model.g_z_all_fun(X_fe[j], self.rk_stage_z(j), Uk, self.p)
            self.add_constraint(gj)
            if opts.irk_representation == IrkRepresentation.INTEGRAL:
                xj = opts.C_irk[0, j + 1] * self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r in range(opts.n_s):
                    xj += opts.C_irk[r + 1, j + 1] * X_fe[r]
                Xk_end += opts.D_irk[j + 1] * X_fe[j]
                self.add_constraint(self.h() * fj - xj)
                self.cost += opts.B_irk[j + 1] * self.h() * qj
            elif (opts.irk_representation
                  in [IrkRepresentation.DIFFERENTIAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
                Xk_end += self.h() * opts.b_irk[j] * self.w[self.ind_v[j]]
                self.add_constraint(fj - self.w[self.ind_v[j]])
                self.cost += opts.b_irk[j] * self.h() * qj

        # continuity condition: end of fe state - final stage state
        if (not opts.right_boundary_point_explicit or
                opts.irk_representation == IrkRepresentation.DIFFERENTIAL):
            self.add_constraint(Xk_end - self.w[self.ind_x[-1]])

        # g_z_all constraint for boundary point and continuity of algebraic variables.
        if not opts.right_boundary_point_explicit and opts.use_fesd and (
                self.fe_idx < opts.Nfe_list[self.ctrl_idx] - 1):
            self.add_constraint(
                model.g_z_switching_fun(self.w[self.ind_x[-1]], self.rk_stage_z(-1), Uk, self.p))

        return

    def create_complementarity(self, x: List[ca.SX], y: ca.SX, sigma: ca.SX, tau: ca.SX) -> None:
        """
        adds complementarity constraints corresponding to (x_i, y) for x_i in x to the FiniteElement.

        :param x: list of ca.SX
        :param y: ca.SX
        :param sigma: smoothing parameter
        :param tau: another smoothing parameter
        """
        opts = self.opts

        n = casadi_length(y)

        if opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode.SCHOLTES_INEQ]:
            # g_comp = ca.diag(y) @ casadi_sum_list([x_i for x_i in x]) - sigma # this works too but is a bit slower.
            g_comp = ca.diag(casadi_sum_list([x_i for x_i in x])) @ y - sigma
            # NOTE: this line should be equivalent but yield different results
            # g_comp = casadi_sum_list([ca.diag(x_i) @ y for x_i in x]) - sigma
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER:
            g_comp = ca.SX.zeros(n, 1)
            for j in range(n):
                for x_i in x:
                    g_comp[j] += x_i[j] + y[j] - ca.sqrt(x_i[j]**2 + y[j]**2 + sigma**2)
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER_IP_AUG:
            if len(x) != 1:
                raise Exception("not supported")
            g_comp = ca.SX.zeros(4 * n, 1)
            # classic FB
            for j in range(n):
                for x_i in x:
                    g_comp[j] += x_i[j] + y[j] - ca.sqrt(x_i[j]**2 + y[j]**2 + sigma**2)
            # augment 1
            for j in range(n):
                for x_i in x:
                    g_comp[j + n] = opts.fb_ip_aug1_weight * (x_i[j] - sigma) * ca.sqrt(tau)
                g_comp[j + 2 * n] = opts.fb_ip_aug1_weight * (y[j] - sigma) * ca.sqrt(tau)
            # augment 2
            for j in range(n):
                for x_i in x:
                    g_comp[j + 3 * n] = opts.fb_ip_aug2_weight * (g_comp[j]) * ca.sqrt(1 + (x_i[j] - y[j])**2)

        n_comp = casadi_length(g_comp)
        if opts.mpcc_mode == MpccMode.SCHOLTES_INEQ:
            lb_comp = -np.inf * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))
        elif opts.mpcc_mode in [
                MpccMode.SCHOLTES_EQ, MpccMode.FISCHER_BURMEISTER,
                MpccMode.FISCHER_BURMEISTER_IP_AUG
        ]:
            lb_comp = 0 * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))

        self.add_constraint(g_comp, lb=lb_comp, ub=ub_comp, index=self.ind_comp)

        return

    def create_complementarity_constraints(self, sigma_p: ca.SX, tau: ca.SX, Uk: ca.SX) -> None:
        opts = self.opts
        # handle path complementarities TODO maintain sparsity?
        # TODO: This needs also to provide a lower bound on a,b in a smart way
        #       however this needs to be done carefully to maintain LICQ
        X_fe = self.X_fe()
        for j in range(opts.n_s):
            z = self.w[self.ind_z[j]]
            stage_comps = self.ocp.g_rk_comp_fun(X_fe[j], Uk, z, self.p, self.model.v_global)  # TODO maybe should include stage z
            a, b = ca.horzsplit(stage_comps)
            self.create_complementarity([a], b, sigma_p, tau)
        if self.fe_idx == opts.N_finite_elements-1:
            ctrl_comps = self.ocp.g_ctrl_comp_fun(Uk, self.p, self.model.v_global)
            a, b = ca.horzsplit(ctrl_comps)
            self.create_complementarity([a], b, sigma_p, tau)

        if not opts.use_fesd:
            for j in range(opts.n_s):
                self.create_complementarity([self.Lambda(stage=j)], self.Theta(stage=j), sigma_p,
                                            tau, Uk)
        elif opts.cross_comp_mode == CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER:
            for j in range(opts.n_s):
                # cross comp with prev_fe
                self.create_complementarity([self.Theta(stage=j)], self.prev_fe.Lambda(stage=-1),
                                            sigma_p, tau)
                for jj in range(opts.n_s):
                    # within fe
                    self.create_complementarity([self.Theta(stage=j)], self.Lambda(stage=jj),
                                                sigma_p, tau)
        elif opts.cross_comp_mode == CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA:
            for j in range(opts.n_s):
                # Note: sum_Lambda contains last stage of prev_fe
                Lambda_list = self.get_Lambdas_incl_last_prev_fe()
                self.create_complementarity(Lambda_list, (self.Theta(stage=j)), sigma_p, tau)
        return

    def step_equilibration(self, sigma_p: ca.SX, tau: ca.SX) -> None:
        opts = self.opts
        # step equilibration only within control stages.
        if not opts.use_fesd:
            return
        if not self.fe_idx > 0:
            return

        prev_fe: FiniteElement = self.prev_fe
        delta_h_ki = self.h() - prev_fe.h()
        if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_MEAN:
            h_fe = opts.terminal_time / (opts.N_stages * opts.Nfe_list[self.ctrl_idx])
            self.cost += opts.rho_h * (self.h() - h_fe)**2
            return
        elif opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA:
            self.cost += opts.rho_h * delta_h_ki**2
            return

        # modes that need nu_k
        eta_k = prev_fe.sum_Lambda() * self.sum_Lambda() + \
                prev_fe.sum_Theta() * self.sum_Theta()
        nu_k = 1
        for jjj in range(casadi_length(eta_k)):
            nu_k = nu_k * eta_k[jjj]

        if opts.step_equilibration == StepEquilibrationMode.L2_RELAXED_SCALED:
            self.cost += opts.rho_h * ca.tanh(nu_k / opts.step_equilibration_sigma) * delta_h_ki**2
        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
            self.cost += opts.rho_h * nu_k * delta_h_ki**2
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT:
            self.add_constraint(nu_k * delta_h_ki)
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT_COMPLEMENTARITY:
            self.create_complementarity([nu_k], delta_h_ki, sigma_p, tau)
        elif opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA_H_COMP:
            self.create_complementarity([ca.SX.zeros()], delta_h_ki, sigma_p, tau)
        # elif opts.step_equilibration == StepEquilibrationMode.DIRECT_TANH:
        #     self.add_constraint(ca.tanh(nu_k)*delta_h_ki)
        return


class NosnocProblem(NosnocFormulationObject):

    def __create_control_stage(self, ctrl_idx, prev_fe):
        # Create control vars
        Uk = ca.SX.sym(f'U_{ctrl_idx}', self.model.dims.n_u)
        self.add_variable(Uk, self.ind_u, self.ocp.lbu, self.ocp.ubu,
                          np.zeros((self.model.dims.n_u,)))

        # Create Finite elements in this control stage
        control_stage = []
        for ii in range(self.opts.Nfe_list[ctrl_idx]):
            fe = FiniteElement(self.opts,
                               self.model,
                               self.ocp,
                               ctrl_idx,
                               fe_idx=ii,
                               prev_fe=prev_fe)
            self._add_finite_element(fe, ctrl_idx)
            control_stage.append(fe)
            prev_fe = fe
        return control_stage

    def __create_primal_variables(self):
        # Initial
        self.fe0 = FiniteElementZero(self.opts, self.model)
        x0 = self.fe0.w[self.fe0.ind_x[0]]
        lambda00 = self.fe0.Lambda()

        # lambda00 is parameter
        self.p = ca.vertcat(self.p, lambda00)
        self.p = ca.vertcat(self.p, x0)

        # v_global
        self.add_variable(self.model.v_global, self.ind_v_global, self.ocp.lbv_global,
                          self.ocp.ubv_global, self.ocp.v_global_guess)

        # Generate control_stages
        prev_fe = self.fe0
        for ii in range(self.opts.N_stages):
            stage = self.__create_control_stage(ii, prev_fe=prev_fe)
            self.stages.append(stage)
            prev_fe = stage[-1]

    def _add_finite_element(self, fe: FiniteElement, ctrl_idx: int):
        w_len = casadi_length(self.w)
        self._add_primal_vector(fe.w, fe.lbw, fe.ubw, fe.w0)

        # update all indices
        self.ind_h.append(fe.ind_h + w_len)
        self.ind_x[ctrl_idx].append(increment_indices(fe.ind_x, w_len))
        self.ind_x_cont[ctrl_idx].append(increment_indices(fe.ind_x[-1], w_len))
        self.ind_v[ctrl_idx].append(increment_indices(fe.ind_v, w_len))
        self.ind_theta[ctrl_idx].append(increment_indices(fe.ind_theta, w_len))
        self.ind_lam[ctrl_idx].append(increment_indices(fe.ind_lam, w_len))
        self.ind_mu[ctrl_idx].append(increment_indices(fe.ind_mu, w_len))
        self.ind_alpha[ctrl_idx].append(increment_indices(fe.ind_alpha, w_len))
        self.ind_lambda_n[ctrl_idx].append(increment_indices(fe.ind_lambda_n, w_len))
        self.ind_lambda_p[ctrl_idx].append(increment_indices(fe.ind_lambda_p, w_len))

    # TODO: can we just use add_variable? It is a bit involved, since index vectors here have different format.
    def _add_primal_vector(self, symbolic: ca.SX, lb: np.array, ub, initial):
        n = casadi_length(symbolic)

        if len(lb) != n or len(ub) != n or len(initial) != n:
            raise Exception(
                f'_add_primal_vector, inconsistent dimension: {symbolic=}, {lb=}, {ub=}, {initial=}'
            )

        self.w = ca.vertcat(self.w, symbolic)
        self.lbw = np.concatenate((self.lbw, lb))
        self.ubw = np.concatenate((self.ubw, ub))
        self.w0 = np.concatenate((self.w0, initial))
        return

    def add_fe_constraints(self, fe: FiniteElement, ctrl_idx: int):
        g_len = casadi_length(self.g)
        self.add_constraint(fe.g, fe.lbg, fe.ubg)
        # constraint indices
        self.ind_comp[ctrl_idx].append(increment_indices(fe.ind_comp, g_len))
        return

    def create_complementarity(self, x: List[ca.SX], y: ca.SX, sigma: ca.SX, tau: ca.SX) -> None:
        """
        adds complementarity constraints corresponding to (x_i, y) for x_i in x to the problem.

        :param x: list of ca.SX
        :param y: ca.SX
        :param sigma: smoothing parameter
        :param tau: another smoothing parameter
        """
        opts = self.opts

        n = casadi_length(y)

        if opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode.SCHOLTES_INEQ]:
            # g_comp = ca.diag(y) @ casadi_sum_list([x_i for x_i in x]) - sigma # this works too but is a bit slower.
            g_comp = ca.diag(casadi_sum_list([x_i for x_i in x])) @ y - sigma
            # NOTE: this line should be equivalent but yield different results
            # g_comp = casadi_sum_list([ca.diag(x_i) @ y for x_i in x]) - sigma
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER:
            g_comp = ca.SX.zeros(n, 1)
            for j in range(n):
                for x_i in x:
                    g_comp[j] += x_i[j] + y[j] - ca.sqrt(x_i[j]**2 + y[j]**2 + sigma**2)
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER_IP_AUG:
            if len(x) != 1:
                raise Exception("not supported")
            g_comp = ca.SX.zeros(4 * n, 1)
            # classic FB
            for j in range(n):
                for x_i in x:
                    g_comp[j] += x_i[j] + y[j] - ca.sqrt(x_i[j]**2 + y[j]**2 + sigma**2)
            # augment 1
            for j in range(n):
                for x_i in x:
                    g_comp[j + n] = opts.fb_ip_aug1_weight * (x_i[j] - sigma) * ca.sqrt(tau)
                g_comp[j + 2 * n] = opts.fb_ip_aug1_weight * (y[j] - sigma) * ca.sqrt(tau)
            # augment 2
            for j in range(n):
                for x_i in x:
                    g_comp[j + 3 * n] = opts.fb_ip_aug2_weight * (g_comp[j]) * ca.sqrt(1 + (x_i[j] - y[j])**2)

        n_comp = casadi_length(g_comp)
        if opts.mpcc_mode == MpccMode.SCHOLTES_INEQ:
            lb_comp = -np.inf * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))
        elif opts.mpcc_mode in [
                MpccMode.SCHOLTES_EQ, MpccMode.FISCHER_BURMEISTER,
                MpccMode.FISCHER_BURMEISTER_IP_AUG
        ]:
            lb_comp = 0 * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))

        self.add_constraint(g_comp, lb=lb_comp, ub=ub_comp, index=self.ind_comp)

        return

    def create_complementarity_constraints(self, sigma_p: ca.SX, tau: ca.SX) -> None:
        # TODO add other complementarity modes here.
        p_global = self.p[self.model.dims.n_p_time_var:self.model.dims.n_p_time_var + self.model.dims.n_p_glob]
        stage_comps = self.ocp.g_global_comp_fun(p_global, self.model.v_global)  # TODO maybe should include stage z
        a, b = ca.horzsplit(stage_comps)
        self.create_complementarity([a], b, sigma_p, tau)
        return

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):

        super().__init__()

        self.model = model
        self.opts = opts
        if ocp is None:
            self.ocp_trivial = True
            ocp = NosnocOcp()
        else:
            self.ocp_trivial = False
        ocp.preprocess_ocp(model)
        self.ocp = ocp

        h_ctrl_stage = opts.terminal_time / opts.N_stages
        self.stages: list[list[FiniteElement]] = []

        # Index vectors of optimization variables
        self.ind_x = create_empty_list_matrix((opts.N_stages,))
        self.ind_x_cont = create_empty_list_matrix((opts.N_stages,))
        self.ind_v = create_empty_list_matrix((opts.N_stages,))
        self.ind_theta = create_empty_list_matrix((opts.N_stages,))
        self.ind_lam = create_empty_list_matrix((opts.N_stages,))
        self.ind_mu = create_empty_list_matrix((opts.N_stages,))
        self.ind_alpha = create_empty_list_matrix((opts.N_stages,))
        self.ind_lambda_n = create_empty_list_matrix((opts.N_stages,))
        self.ind_lambda_p = create_empty_list_matrix((opts.N_stages,))

        self.ind_u = []
        self.ind_h = []
        self.ind_v_global = []

        # Index vectors within constraints g
        self.ind_comp = create_empty_list_matrix((opts.N_stages,))

        # setup parameters, lambda00 is added later:
        sigma_p = ca.SX.sym('sigma_p')  # homotopy parameter
        tau = ca.SX.sym('tau')  # homotopy parameter
        self.p = ca.vertcat(casadi_vertcat_list(model.p_ctrl_stages), sigma_p, tau)

        # Generate all the variables we need
        self.__create_primal_variables()

        fe: FiniteElement
        stage: List[FiniteElement]
        if opts.time_freezing:
            t0 = model.t_fun(self.fe0.w[self.fe0.ind_x[-1]])

        for ctrl_idx, stage in enumerate(self.stages):
            Uk = self.w[self.ind_u[ctrl_idx]]
            for _, fe in enumerate(stage):

                # 1) Stewart Runge-Kutta discretization
                fe.forward_simulation(ocp, Uk)

                # 2) Complementarity Constraints
                fe.create_complementarity_constraints(sigma_p, tau, Uk)

                # 3) Step Equilibration
                fe.step_equilibration(sigma_p, tau)

                # 4) add cost and constraints from FE to problem
                self.cost += fe.cost
                self.add_fe_constraints(fe, ctrl_idx)

            if opts.use_fesd and opts.equidistant_control_grid:
                self.add_constraint(sum([fe.h() for fe in stage]) - h_ctrl_stage)

            if opts.time_freezing and opts.equidistant_control_grid:
                # TODO: make t0 dynamic (since now it needs to be 0!)
                t_now = opts.terminal_time / opts.N_stages * (ctrl_idx + 1) + t0
                Xk_end = stage[-1].w[stage[-1].ind_x[-1]]
                self.add_constraint(
                    model.t_fun(Xk_end) - t_now, [-opts.time_freezing_tolerance],
                    [opts.time_freezing_tolerance])

        # Create global complementarities
        self.create_complementarity_constraints(sigma_p, tau)

        # Scalar-valued complementarity residual
        if opts.use_fesd:
            J_comp = 0.0
            for fe in flatten(self.stages):
                sum_abs_lam = casadi_sum_list(
                    [ca.fabs(lam) for lam in fe.get_Lambdas_incl_last_prev_fe()])
                sum_abs_theta = casadi_sum_list([ca.fabs(t) for t in fe.get_Theta_list()])
                J_comp += ca.sum1(ca.diag(sum_abs_theta) @ sum_abs_lam)
        else:
            J_comp = casadi_sum_list([
                model.std_compl_res_fun(fe.rk_stage_z(j), fe.p)
                for j in range(opts.n_s)
                for fe in flatten(self.stages)
            ])

        # terminal constraint and cost
        # NOTE: this was evaluated at Xk_end (expression for previous state before)
        # which should be worse for convergence.
        x_terminal = self.w[self.ind_x[-1][-1][-1]]
        g_terminal = ocp.g_terminal_fun(x_terminal, model.p_ctrl_stages[-1], model.v_global)
        self.add_constraint(g_terminal)
        self.cost += ocp.f_q_T_fun(x_terminal, model.p_ctrl_stages[-1], model.v_global)

        # Terminal numerical time
        if opts.N_stages > 1 and opts.use_fesd:
            all_h = [fe.h() for stage in self.stages for fe in stage]
            self.add_constraint(sum(all_h) - opts.terminal_time)

        # CasADi Functions
        self.cost_fun = ca.Function('cost_fun', [self.w], [self.cost])
        self.comp_res = ca.Function('comp_res', [self.w, self.p], [J_comp])
        self.g_fun = ca.Function('g_fun', [self.w, self.p], [self.g])

        # copy original w0
        self.w0_original = self.w0.copy()

        # LEAST_SQUARES reformulation
        if opts.constraint_handling == ConstraintHandling.LEAST_SQUARES:
            self.g_lsq = copy(self.g)
            for ii in range(casadi_length(self.g)):
                if self.lbg[ii] != 0.0:
                    raise Exception(
                        f"least_squares constraint handling only supported if all lbg, ubg == 0.0, got {self.lbg[ii]=}, {self.ubg[ii]=}, {self.g[ii]=}"
                    )
                self.cost += self.g[ii]**2
            self.g = ca.SX([])
            self.lbg = np.array([])
            self.ubg = np.array([])

    def print(self):
        # constraints
        print("lbg\t\t ubg\t\t g_expr")
        for i in range(len(self.lbg)):
            print(f"{self.lbg[i]:7} \t {self.ubg[i]:7} \t {self.g[i]}")
        # variables and bounds
        print("\nw \t\t\t w0 \t\t lbw \t\t ubw:")
        for i in range(len(self.lbw)):
            print(
                f"{self.w[i].name():<15} \t {self.w0[i]:.2e} \t {self.lbw[i]:7} \t {self.ubw[i]:.2e}"
            )
        # cost
        print(f"\ncost:\n{self.cost}")

    def is_sim_problem(self):
        if self.model.dims.n_u != 0:
            return False
        if self.opts.N_stages != 1:
            return False
        if not self.ocp_trivial:
            return False
        return True
