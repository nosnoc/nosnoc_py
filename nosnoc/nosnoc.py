from typing import Optional, List
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field

import numpy as np
from casadi import SX, vertcat, horzcat, sum1, inf, Function, diag, nlpsol, fabs, tanh, mmin, transpose, fmax, fmin, exp

from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule
from nosnoc.utils import casadi_length, print_casadi_vector, casadi_vertcat_list, casadi_sum_list, flatten_layer, flatten, increment_indices
from nosnoc.rk_utils import rk4_on_timegrid


class NosnocModel:
    r"""
    \dot{x} \in f_i(x, u) if x(t) in R_i \subset \R^{n_x}

    with R_i = {x \in \R^{n_x} | diag(S_i,\dot) * c(x) > 0}

    where S_i denotes the rows of S.
    """

    # TODO: extend docu for n_sys > 1
    # NOTE: n_sys is needed decoupled systems: see FESD: "Remark on Cartesian products of Filippov systems"

    def __init__(self,
                 x: SX,
                 F: List[SX],
                 c: List[SX],
                 S: List[np.ndarray],
                 x0: np.ndarray,
                 u: SX = SX.sym('u_dummy', 0, 1),
                 name: str = 'nosnoc'):
        self.x: SX = x
        self.F: List[SX] = F
        self.c: List[SX] = c
        self.S: List[np.ndarray] = S
        self.x0: np.ndarray = x0
        self.u: SX = u
        self.name: str = name

        self.dims: NosnocDims = None

    def preprocess_model(self, opts: NosnocOpts):
        # detect dimensions
        nx = casadi_length(self.x)
        nu = casadi_length(self.u)
        n_sys = len(self.F)
        n_c_sys = [casadi_length(self.c[i]) for i in range(n_sys)]
        n_f_sys = [self.F[i].shape[1] for i in range(n_sys)]

        self.dims = NosnocDims(nx=nx, nu=nu, n_sys=n_sys, n_c_sys=n_c_sys, n_f_sys=n_f_sys)

        g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(n_sys)]
        g_Stewart = casadi_vertcat_list(g_Stewart_list)

        # create dummy finite element - only use first stage
        fe = FiniteElement(opts, self, ctrl_idx=0, fe_idx=0, prev_fe=None)

        # setup upsilon
        upsilon = []
        if opts.pss_mode == PssMode.STEP:
            for ii in range(self.dims.n_sys):
                upsilon_temp = []
                S_temp = self.S[ii]
                for j in range(len(S_temp)):
                    upsilon_ij = 1
                    for k in range(len(S_temp[0, :])):
                        # create multiafine term
                        if S_temp[j, k] != 0:
                            upsilon_ij = upsilon_ij * (0.5 * (1 - S_temp[j, k]) +
                                                       S_temp[j, k] * fe.w[fe.ind_alpha[0][ii]][k])
                    upsilon_temp = vertcat(upsilon_temp, upsilon_ij)
                upsilon = horzcat(upsilon, upsilon_temp)

        # start empty
        g_lift = SX.zeros((0, 1))
        g_switching = SX.zeros((0, 1))
        g_convex = SX.zeros((0, 1))  # equation for the convex multiplers 1 = e' \theta
        lambda00_expr = SX.zeros(0, 0)
        std_compl_res = SX.zeros(1)  # residual of standard complementarity

        z = fe.rk_stage_z(0)

        # Reformulate the Filippov ODE into a DCS
        f_x = SX.zeros((nx, 1))

        if opts.pss_mode == PssMode.STEWART:
            for ii in range(n_sys):
                f_x = f_x + self.F[ii] @ fe.w[fe.ind_theta[0][ii]]
                g_switching = vertcat(
                    g_switching,
                    g_Stewart_list[ii] - fe.w[fe.ind_lam[0][ii]] + fe.w[fe.ind_mu[0][ii]])
                g_convex = vertcat(g_convex, sum1(fe.w[fe.ind_theta[0][ii]]) - 1)
                std_compl_res += fabs(fe.w[fe.ind_lam[0][ii]].T @ fe.w[fe.ind_theta[0][ii]])
                lambda00_expr = vertcat(lambda00_expr,
                                        g_Stewart_list[ii] - mmin(g_Stewart_list[ii]))
        elif opts.pss_mode == PssMode.STEP:
            for ii in range(n_sys):
                f_x = f_x + self.F[ii] @ upsilon[:, ii]
                g_switching = vertcat(
                    g_switching,
                    self.c[ii] - fe.w[fe.ind_lambda_p[0][ii]] + fe.w[fe.ind_lambda_n[0][ii]])
                std_compl_res += transpose(fe.w[fe.ind_lambda_n[0][ii]]) @ fe.w[fe.ind_alpha[0][ii]]
                std_compl_res += transpose(fe.w[fe.ind_lambda_p[0][ii]]) @ (
                    np.ones(n_c_sys[ii]) - fe.w[fe.ind_alpha[0][ii]])
                lambda00_expr = vertcat(lambda00_expr, -fmin(self.c[ii], 0), fmax(self.c[ii], 0))

        mu00_stewart = casadi_vertcat_list([mmin(g_Stewart_list[ii]) for ii in range(n_sys)])

        # collect all algebraic equations
        g_z_all = vertcat(g_switching, g_convex, g_lift)  # g_lift_forces

        # CasADi functions for indicator and region constraint functions
        self.z = z
        self.g_Stewart_fun = Function('g_Stewart_fun', [self.x], [g_Stewart])
        self.c_fun = Function('c_fun', [self.x], [casadi_vertcat_list(self.c)])

        # dynamics
        self.f_x_fun = Function('f_x_fun', [self.x, z, self.u], [f_x])

        # lp kkt conditions without bilinear complementarity terms
        self.g_z_switching_fun = Function('g_z_switching_fun', [self.x, z, self.u], [g_switching])
        self.g_z_all_fun = Function('g_z_all_fun', [self.x, z, self.u], [g_z_all])
        self.lambda00_fun = Function('lambda00_fun', [self.x], [lambda00_expr])

        self.std_compl_res_fun = Function('std_compl_res_fun', [z], [std_compl_res])
        self.mu00_stewart_fun = Function('mu00_stewart_fun', [self.x], [mu00_stewart])

    def add_smooth_step_representation(self, smoothing_parameter=1e-1):
        dims = self.dims

        # smooth step function
        y = SX.sym('y')
        smooth_step_fun = Function('smooth_step_fun', [y],
                                   [(tanh(smoothing_parameter * y) + 1) / 2])

        lambda_smooth = []
        g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(dims.n_sys)]

        theta_list = [SX.zeros(nf) for nf in dims.n_f_sys]
        mu_smooth_list = []
        f_x_smooth = SX.zeros((dims.nx, 1))
        for s in range(dims.n_sys):
            n_c: int = dims.n_c_sys[s]
            alpha_expr_s = casadi_vertcat_list([smooth_step_fun(self.c[s][i]) for i in range(n_c)])

            min_in = SX.sym('min_in', dims.n_f_sys[s])
            min_out = sum1(casadi_vertcat_list([min_in[i]*exp(-smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))])) / \
                      sum1(casadi_vertcat_list([exp(-smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))]))
            smooth_min_fun = Function('smooth_min_fun', [min_in], [min_out])
            mu_smooth_list.append(-smooth_min_fun(g_Stewart_list[s]))
            lambda_smooth = vertcat(lambda_smooth,
                                    g_Stewart_list[s] - smooth_min_fun(g_Stewart_list[s]))

            for i in range(dims.n_f_sys[s]):
                n_Ri = sum(np.abs(self.S[s][i, :]))
                theta_list[s][i] = 2**(n_c - n_Ri)
                for j in range(n_c):
                    theta_list[s][i] *= ((1 - self.S[s][i, j]) / 2 +
                                         self.S[s][i, j] * alpha_expr_s[j])
            f_x_smooth += self.F[s] @ theta_list[s]

        theta_smooth = casadi_vertcat_list(theta_list)
        mu_smooth = casadi_vertcat_list(mu_smooth_list)

        self.f_x_smooth_fun = Function('f_x_smooth_fun', [self.x], [f_x_smooth])
        self.theta_smooth_fun = Function('theta_smooth_fun', [self.x], [theta_smooth])
        self.mu_smooth_fun = Function('mu_smooth_fun', [self.x], [mu_smooth])
        self.lambda_smooth_fun = Function('lambda_smooth_fun', [self.x], [lambda_smooth])


class NosnocOcp:
    """
    allows to specify

    1) constraints of the form:
    lbu <= u <= ubu
    g_terminal(x_terminal) = 0

    2) cost of the form:
    f_q(x, u)  -- integrated over the time horizon
    +
    f_q_T(x_terminal) -- evaluated at the end
    """

    def __init__(self,
                 lbu: np.ndarray = np.ones((0,)),
                 ubu: np.ndarray = np.ones((0,)),
                 f_q: SX = SX.zeros(1),
                 f_q_T: SX = SX.zeros(1),
                 g_terminal: SX = SX.zeros(0)):
        self.lbu: np.ndarray = lbu
        self.ubu: np.ndarray = ubu
        self.f_q: SX = f_q
        self.f_q_T: SX = f_q_T
        self.g_terminal: SX = g_terminal

    def preprocess_ocp(self, x: SX, u: SX):
        self.g_terminal_fun = Function('g_terminal_fun', [x], [self.g_terminal])
        self.f_q_fun = Function('f_q_fun', [x, u], [self.f_q])


@dataclass
class NosnocDims:
    """
    detected automatically
    """
    nx: int = 0
    nu: int = 0
    n_sys: int = 0
    n_c_sys: list = field(default_factory=list)
    n_f_sys: list = field(default_factory=list)


class NosnocFormulationObject(ABC):

    @abstractmethod
    def __init__(self):
        self.w: SX = SX([])
        self.w0: np.array = np.array([])
        self.lbw: np.array = np.array([])
        self.ubw: np.array = np.array([])

        self.g: SX = SX([])
        self.lbg: np.array = np.array([])
        self.ubg: np.array = np.array([])

        self.cost: SX = SX.zeros(1)

    def __repr__(self):
        return repr(self.__dict__)

    def add_variable(self,
                     symbolic: SX,
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

        self.w = vertcat(self.w, symbolic)
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

    def add_constraint(self, symbolic: SX, lb=None, ub=None):
        n = casadi_length(symbolic)
        if lb is None:
            lb = np.zeros((n,))
        if ub is None:
            ub = np.zeros((n,))
        if len(lb) != n or len(ub) != n:
            raise Exception(f'add_constraint, inconsistent dimension: {symbolic=}, {lb=}, {ub=}')

        self.g = vertcat(self.g, symbolic)
        self.lbg = np.concatenate((self.lbg, lb))
        self.ubg = np.concatenate((self.ubg, ub))

        return


class FiniteElementBase(NosnocFormulationObject):

    def Lambda(self, stage=slice(None), sys=slice(None)):
        return vertcat(self.w[flatten(self.ind_lam[stage][sys])],
                       self.w[flatten(self.ind_lambda_n[stage][sys])],
                       self.w[flatten(self.ind_lambda_p[stage][sys])])

    def sum_Lambda(self, sys=slice(None)):
        Lambdas = [self.Lambda(stage=ii, sys=sys) for ii in range(len(self.ind_lam))]
        Lambdas.append(self.prev_fe.Lambda(
            stage=-1, sys=sys))  # Include the last finite element's last stage lambda
        return casadi_sum_list(Lambdas)


class FiniteElementZero(FiniteElementBase):

    def __init__(self, opts: NosnocOpts, model: NosnocModel):
        super().__init__()
        dims = model.dims
        self.n_rkstages = 1

        self.ind_x = np.empty((1, 0), dtype=int).tolist()
        self.ind_lam = np.empty((self.n_rkstages, dims.n_sys, 0), dtype=int).tolist()
        self.ind_lambda_n = np.empty((self.n_rkstages, dims.n_sys, 0), dtype=int).tolist()
        self.ind_lambda_p = np.empty((self.n_rkstages, dims.n_sys, 0), dtype=int).tolist()

        # NOTE: bounds are actually not used, maybe rewrite without add_vairable
        # X0
        self.add_variable(SX.sym('X0', dims.nx), self.ind_x, model.x0, model.x0, model.x0, 0)

        # lambda00
        if opts.pss_mode == PssMode.STEWART:
            for ij in range(dims.n_sys):
                self.add_variable(SX.sym(f'lambda00_{ij+1}', dims.n_f_sys[ij]), self.ind_lam,
                                  -inf * np.ones(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_f_sys[ij]), 0, ij)
        elif opts.pss_mode == PssMode.STEP:
            for ij in range(dims.n_sys):
                self.add_variable(SX.sym(f'lambda00_n_{ij+1}', dims.n_c_sys[ij]), self.ind_lambda_n,
                                  -inf * np.ones(dims.n_c_sys[ij]), inf * np.ones(dims.n_c_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_c_sys[ij]), 0, ij)
                self.add_variable(SX.sym(f'lambda00_p_{ij+1}', dims.n_c_sys[ij]), self.ind_lambda_p,
                                  -inf * np.ones(dims.n_c_sys[ij]), inf * np.ones(dims.n_c_sys[ij]),
                                  opts.init_lambda * np.ones(dims.n_c_sys[ij]), 0, ij)


class FiniteElement(FiniteElementBase):

    def __init__(self,
                 opts: NosnocOpts,
                 model: NosnocModel,
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

        dims = self.model.dims

        # right boundary
        create_right_boundary_point = (opts.use_fesd and not opts.right_boundary_point_explicit and
                                       fe_idx < opts.Nfe_list[ctrl_idx] - 1)
        end_allowance = 1 if create_right_boundary_point else 0

        # Initialze index vectors. Note ind_x contains an extra element
        # in order to store the end variables
        # TODO: add helper: create_list_mat(n_s+1, 0)
        # TODO: if irk tableau contains end point, we should only use n_s state variables!
        self.ind_x = np.empty((n_s + 1, 0), dtype=int).tolist()
        if opts.irk_representation == IrkRepresentation.DIFFERENTIAL and not opts.lift_irk_differential:
            self.ind_x = np.empty((1, 0), dtype=int).tolist()
        self.ind_v = np.empty((n_s, 0), dtype=int).tolist()
        self.ind_theta = np.empty((n_s, dims.n_sys, 0), dtype=int).tolist()
        self.ind_lam = np.empty((n_s + end_allowance, dims.n_sys, 0), dtype=int).tolist()
        self.ind_mu = np.empty((n_s + end_allowance, dims.n_sys, 0), dtype=int).tolist()
        self.ind_alpha = np.empty((n_s, dims.n_sys, 0), dtype=int).tolist()
        self.ind_lambda_n = np.empty((n_s + end_allowance, dims.n_sys, 0), dtype=int).tolist()
        self.ind_lambda_p = np.empty((n_s + end_allowance, dims.n_sys, 0), dtype=int).tolist()
        self.ind_h = []

        self.prev_fe: FiniteElementBase = prev_fe

        # create variables
        h = SX.sym(f'h_{ctrl_idx}_{fe_idx}')
        h_ctrl_stage = opts.terminal_time / opts.N_stages
        h0 = np.array([h_ctrl_stage / np.array(opts.Nfe_list[ctrl_idx])])
        ubh = (1 + opts.gamma_h) * h0
        lbh = (1 - opts.gamma_h) * h0
        self.add_step_size_variable(h, lbh, ubh, h0)

        # RK stage stuff
        for ii in range(opts.n_s):
            # state / state derivative variables
            if opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
                self.add_variable(SX.sym(f'V_{ctrl_idx}_{fe_idx}_{ii+1}', dims.nx),
                                  self.ind_v, -inf * np.ones(dims.nx), inf * np.ones(dims.nx),
                                  np.zeros(dims.nx), ii)
            if opts.irk_representation == IrkRepresentation.INTEGRAL or opts.lift_irk_differential:
                self.add_variable(SX.sym(f'X_{ctrl_idx}_{fe_idx}_{ii+1}', dims.nx), self.ind_x,
                                  -inf * np.ones(dims.nx), inf * np.ones(dims.nx), model.x0, ii)
            # algebraic variables
            if opts.pss_mode == PssMode.STEWART:
                # add thetas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'theta_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_theta, np.zeros(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_theta * np.ones(dims.n_f_sys[ij]), ii, ij)
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, np.zeros(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_f_sys[ij]), ii, ij)
                # add mu
                for ij in range(dims.n_sys):
                    self.add_variable(SX.sym(f'mu_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', 1),
                                      self.ind_mu, -inf * np.ones(1), inf * np.ones(1),
                                      opts.init_mu * np.ones(1), ii, ij)
            elif opts.pss_mode == PssMode.STEP:
                # add alpha
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'alpha_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_c_sys[ij]),
                        self.ind_alpha, np.zeros(dims.n_c_sys[ij]), np.ones(dims.n_c_sys[ij]),
                        opts.init_theta * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_n
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_n_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_n, np.zeros(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_p, np.zeros(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]), opts.init_mu * np.ones(dims.n_c_sys[ij]),
                        ii, ij)

        # Add right boundary points if needed
        if create_right_boundary_point:
            if opts.pss_mode == PssMode.STEWART:
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_end_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, np.zeros(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_f_sys[ij]), opts.n_s, ij)
                # add mu
                for ij in range(dims.n_sys):
                    self.add_variable(SX.sym(f'mu_{ctrl_idx}_{fe_idx}_end_{ij+1}', 1),
                                      self.ind_mu, -inf * np.ones(1), inf * np.ones(1),
                                      opts.init_mu * np.ones(1), opts.n_s, ij)
            elif opts.pss_mode == PssMode.STEP:
                # add lambda_n
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_n_{ctrl_idx}_{fe_idx}_end_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_n, np.zeros(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), opts.n_s, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_end_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_p, np.zeros(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]), opts.init_mu * np.ones(dims.n_c_sys[ij]),
                        opts.n_s, ij)
        # add final X variables
        self.add_variable(SX.sym(f'X_end_{ctrl_idx}_{fe_idx+1}', dims.nx), self.ind_x,
                          -inf * np.ones(dims.nx), inf * np.ones(dims.nx), model.x0, -1)

    def add_step_size_variable(self, symbolic: SX, lb: float, ub: float, initial: float):
        self.ind_h = casadi_length(self.w)
        self.w = vertcat(self.w, symbolic)

        self.lbw = np.append(self.lbw, lb)
        self.ubw = np.append(self.ubw, ub)
        self.w0 = np.append(self.w0, initial)
        return

    def rk_stage_z(self, stage) -> SX:
        idx = np.concatenate((flatten(self.ind_theta[stage]), flatten(self.ind_lam[stage]),
                              flatten(self.ind_mu[stage]), flatten(self.ind_alpha[stage]),
                              flatten(self.ind_lambda_n[stage]), flatten(self.ind_lambda_p[stage])))
        return self.w[idx]

    def Theta(self, stage=slice(None), sys=slice(None)) -> SX:
        return vertcat(
            self.w[flatten(self.ind_theta[stage][sys])],
            self.w[flatten(self.ind_alpha[stage][sys])],
            np.ones(len(flatten(self.ind_alpha[stage][sys]))) -
            self.w[flatten(self.ind_alpha[stage][sys])])

    def sum_Theta(self) -> SX:
        Thetas = [self.Theta(stage=ii) for ii in range(len(self.ind_theta))]
        return casadi_sum_list(Thetas)

    def h(self) -> SX:
        return self.w[self.ind_h]

    def forward_simulation(self, ocp: NosnocOcp, Uk: SX) -> None:
        opts = self.opts
        model = self.model

        if opts.irk_representation == IrkRepresentation.INTEGRAL:
            X_ki = [self.w[x_kij] for x_kij in self.ind_x]
            Xk_end = opts.D_irk[0] * self.prev_fe.w[self.prev_fe.ind_x[-1]]

        if opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
            X_ki = []
            for j in range(opts.n_s):  # Ignore continuity vars
                x_temp = self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r in range(opts.n_s):
                    x_temp += self.h() * opts.A_irk[j, r] * self.w[self.ind_v[r]]
                if opts.lift_irk_differential:
                    X_ki.append(self.w[self.ind_x[j]])
                    self.add_constraint(self.w[self.ind_x[j]] - x_temp)
                else:
                    X_ki.append(x_temp)
            X_ki.append(self.w[self.ind_x[-1]])
            Xk_end = self.prev_fe.w[self.prev_fe.ind_x[-1]]  # initialize

        for j in range(opts.n_s):
            # Dynamics excluding complementarities
            fj = model.f_x_fun(X_ki[j], self.rk_stage_z(j), Uk)
            gj = model.g_z_all_fun(X_ki[j], self.rk_stage_z(j), Uk)
            qj = ocp.f_q_fun(X_ki[j], Uk)
            if opts.irk_representation == IrkRepresentation.INTEGRAL:
                xp = opts.C_irk[0, j + 1] * self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r, x in enumerate(X_ki[:-1]):
                    xp += opts.C_irk[r + 1, j + 1] * x
                Xk_end += opts.D_irk[j + 1] * X_ki[j]
                self.add_constraint(self.h() * fj - xp)
                self.cost += opts.B_irk[j + 1] * self.h() * qj
            elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
                Xk_end += self.h() * opts.b_irk[j] * self.w[self.ind_v[j]]
                self.add_constraint(fj - self.w[self.ind_v[j]])
                self.cost += opts.b_irk[j] * self.h() * qj
            self.add_constraint(gj)
        # continuity condition: end of fe state - final stage state
        self.add_constraint(Xk_end - self.w[self.ind_x[-1]])

        # g_z_all constraint for boundary point and continuity of algebraic variables.
        if not opts.right_boundary_point_explicit and opts.use_fesd and (
                self.fe_idx < opts.Nfe_list[self.ctrl_idx] - 1):
            self.add_constraint(
                model.g_z_switching_fun(self.w[self.ind_x[-1]], self.rk_stage_z(-1), Uk))

        return

    def create_complementarity_constraints(self, sigma_p: SX) -> None:
        opts = self.opts
        dims = self.model.dims
        if not opts.use_fesd:
            g_cross_comp = casadi_vertcat_list(
                [diag(self.Lambda(stage=j)) @ self.Theta(stage=j) for j in range(opts.n_s)])

        elif opts.cross_comp_mode == CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER:
            # complement within fe
            g_cross_comp = casadi_vertcat_list([
                diag(self.Theta(stage=j, sys=r)) @ self.Lambda(stage=jj, sys=r)
                for r in range(dims.n_sys) for j in range(opts.n_s) for jj in range(opts.n_s)
            ])
            # complement with end of previous fe
            g_cross_comp = casadi_vertcat_list([g_cross_comp] + [
                diag(self.Theta(stage=j, sys=r)) @ self.prev_fe.Lambda(stage=-1, sys=r)
                for r in range(dims.n_sys)
                for j in range(opts.n_s)
            ])
        elif opts.cross_comp_mode == CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA:
            # Note: sum_Lambda contains last stage of prev_fe
            g_cross_comp = casadi_vertcat_list([
                diag(self.Theta(stage=j, sys=r)) @ self.sum_Lambda(sys=r)
                for r in range(dims.n_sys)
                for j in range(opts.n_s)
            ])

        n_cross_comp = casadi_length(g_cross_comp)
        g_cross_comp = g_cross_comp - sigma_p
        g_cross_comp_ub = 0 * np.ones((n_cross_comp,))
        if opts.mpcc_mode == MpccMode.SCHOLTES_INEQ:
            g_cross_comp_lb = -np.inf * np.ones((n_cross_comp,))
        elif opts.mpcc_mode == MpccMode.SCHOLTES_EQ:
            g_cross_comp_lb = 0 * np.ones((n_cross_comp,))

        self.add_constraint(g_cross_comp, lb=g_cross_comp_lb, ub=g_cross_comp_ub)

        return

    def step_equilibration(self) -> None:
        opts = self.opts
        prev_fe = self.prev_fe
        if opts.use_fesd and self.fe_idx > 0:  # step equilibration only within control stages.
            delta_h_ki = self.h() - prev_fe.h()
            if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_MEAN:
                h_fe = opts.terminal_time / (opts.N_stages * opts.Nfe_list[self.ctrl_idx])
                self.cost += opts.rho_h * (self.h() - h_fe)**2
            elif opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA:
                self.cost += opts.rho_h * delta_h_ki**2
            elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED_SCALED:
                eta_k = prev_fe.sum_Lambda() * self.sum_Lambda() + \
                        prev_fe.sum_Theta() * self.sum_Theta()
                nu_k = 1
                for jjj in range(casadi_length(eta_k)):
                    nu_k = nu_k * eta_k[jjj]
                self.cost += opts.rho_h * tanh(nu_k / opts.step_equilibration_sigma) * delta_h_ki**2
            elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
                eta_k = prev_fe.sum_Lambda() * self.sum_Lambda() + \
                        prev_fe.sum_Theta() * self.sum_Theta()
                nu_k = 1
                for jjj in range(casadi_length(eta_k)):
                    nu_k = nu_k * eta_k[jjj]
                self.cost += opts.rho_h * nu_k * delta_h_ki**2
        return


class NosnocProblem(NosnocFormulationObject):

    def __create_control_stage(self, ctrl_idx, prev_fe):
        # Create control vars
        Uk = SX.sym(f'U_{ctrl_idx}', self.model.dims.nu)
        self.add_variable(Uk, self.ind_u, self.ocp.lbu, self.ocp.ubu, np.zeros(
            (self.model.dims.nu,)))

        # Create Finite elements in this control stage
        control_stage = []
        for ii in range(self.opts.Nfe_list[ctrl_idx]):
            fe = FiniteElement(self.opts, self.model, ctrl_idx, fe_idx=ii, prev_fe=prev_fe)
            self._add_finite_element(fe)
            control_stage.append(fe)
            prev_fe = fe
        return control_stage

    def __create_primal_variables(self):
        # Initial
        self.fe0 = FiniteElementZero(self.opts, self.model)

        # lambda00 is parameter
        self.p = vertcat(self.p, self.fe0.Lambda())

        # X0 is variable
        self.add_variable(self.fe0.w[self.fe0.ind_x[0]], self.ind_x,
                          self.fe0.lbw[self.fe0.ind_x[0]], self.fe0.ubw[self.fe0.ind_x[0]],
                          self.fe0.w0[self.fe0.ind_x[0]])

        # Generate control_stages
        prev_fe = self.fe0
        for ii in range(self.opts.N_stages):
            stage = self.__create_control_stage(ii, prev_fe=prev_fe)
            self.stages.append(stage)
            prev_fe = stage[-1]

    def _add_finite_element(self, fe: FiniteElement):
        w_len = casadi_length(self.w)
        self._add_primal_vector(fe.w, fe.lbw, fe.ubw, fe.w0)

        # update all indices
        self.ind_h.append(fe.ind_h + w_len)
        self.ind_x.append(increment_indices(fe.ind_x, w_len))
        self.ind_x_cont.append(increment_indices(fe.ind_x[-1], w_len))
        self.ind_v.append(increment_indices(fe.ind_v, w_len))
        self.ind_theta.append(increment_indices(fe.ind_theta, w_len))
        self.ind_lam.append(increment_indices(fe.ind_lam, w_len))
        self.ind_mu.append(increment_indices(fe.ind_mu, w_len))
        self.ind_alpha.append(increment_indices(fe.ind_alpha, w_len))
        self.ind_lambda_n.append(increment_indices(fe.ind_lambda_n, w_len))
        self.ind_lambda_p.append(increment_indices(fe.ind_lambda_p, w_len))

    # TODO: can we just use add_variable? It is a bit involved, since index vectors here have different format.
    def _add_primal_vector(self, symbolic: SX, lb: np.array, ub, initial):
        n = casadi_length(symbolic)

        if len(lb) != n or len(ub) != n or len(initial) != n:
            raise Exception(
                f'_add_primal_vector, inconsistent dimension: {symbolic=}, {lb=}, {ub=}, {initial=}'
            )

        self.w = vertcat(self.w, symbolic)
        self.lbw = np.concatenate((self.lbw, lb))
        self.ubw = np.concatenate((self.ubw, ub))
        self.w0 = np.concatenate((self.w0, initial))
        return

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):

        super().__init__()

        self.model = model
        self.ocp = ocp
        self.opts = opts

        h_ctrl_stage = opts.terminal_time / opts.N_stages
        self.stages: list[list[FiniteElementBase]] = []

        # Index vectors
        self.ind_x = []
        self.ind_x_cont = []
        self.ind_v = []
        self.ind_theta = []
        self.ind_lam = []
        self.ind_mu = []
        self.ind_alpha = []
        self.ind_lambda_n = []
        self.ind_lambda_p = []
        self.ind_u = []
        self.ind_h = []

        # setup parameters, lambda00 is added later:
        sigma_p = SX.sym('sigma_p')  # homotopy parameter
        self.p = sigma_p

        # Generate all the variables we need
        self.__create_primal_variables()

        fe: FiniteElement
        stage: List[FiniteElementBase]
        for k, stage in enumerate(self.stages):
            Uk = self.w[self.ind_u[k]]
            for _, fe in enumerate(stage):

                # 1) Stewart Runge-Kutta discretization
                fe.forward_simulation(ocp, Uk)

                # 2) Complementarity Constraints
                fe.create_complementarity_constraints(sigma_p)

                # 3) Step Equilibration
                fe.step_equilibration()

                # 4) add cost and constraints from FE to problem
                self.cost += fe.cost
                self.add_constraint(fe.g, fe.lbg, fe.ubg)

            if opts.use_fesd and opts.equidistant_control_grid:
                self.add_constraint(sum([fe.h() for fe in stage]) - h_ctrl_stage)

        # Scalar-valued complementarity residual
        if opts.use_fesd:
            J_comp = sum1(diag(fe.sum_Theta()) @ fe.sum_Lambda())
        else:
            J_comp = casadi_sum_list([
                model.std_compl_res_fun(fe.rk_stage_z(j))
                for j in range(opts.n_s)
                for fe in flatten(self.stages)
            ])

        # terminal constraint
        # NOTE: this was evaluated at Xk_end (expression for previous state before) which should be worse for convergence.
        g_terminal = ocp.g_terminal_fun(self.w[self.ind_x[-1][-1]])
        self.add_constraint(g_terminal)

        # Terminal numerical time
        if opts.N_stages > 1 and opts.use_fesd:
            all_h = [fe.h() for stage in self.stages for fe in stage]
            self.add_constraint(sum(all_h) - opts.terminal_time)

        # CasADi Functions
        self.cost_fun = Function('cost_fun', [self.w], [self.cost])
        self.comp_res = Function('comp_res', [self.w, self.p], [J_comp])
        self.g_fun = Function('g_fun', [self.w, self.p], [self.g])

    def print(self):
        print("g:")
        print_casadi_vector(self.g)
        print(f"lbg, ubg\n{np.vstack((self.lbg, self.ubg)).T}")
        print("w:")
        print_casadi_vector(self.w)
        print(f"lbw, ubw\n{np.vstack((self.lbw, self.ubw)).T}")
        print("w0")
        for xx in self.w0:
            print(xx)
        print(f"cost:\n{self.cost}")


def get_results_from_primal_vector(prob: NosnocProblem, w_opt: np.ndarray) -> dict:
    opts = prob.opts

    results = dict()
    results["x_out"] = w_opt[prob.ind_x[-1][-1]]
    # TODO: improve naming here?
    results["x_list"] = [w_opt[ind] for ind in prob.ind_x_cont]

    ind_x_all = [prob.ind_x[0]] + [ind for ind_list in prob.ind_x[1:] for ind in ind_list]
    results["x_all_list"] = [w_opt[ind] for ind in ind_x_all]

    results["u_list"] = [w_opt[ind] for ind in prob.ind_u]
    results["v_list"] = [w_opt[ind] for ind in prob.ind_v]
    results["theta_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_theta]
    results["lambda_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_lam]
    results["mu_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_mu]

    # if opts.pss_mode == PssMode.STEP:
    results["alpha_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_alpha]
    results["lambda_n_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_lambda_n]
    results["lambda_p_list"] = [w_opt[flatten_layer(ind)] for ind in prob.ind_lambda_p]

    if opts.use_fesd:
        time_steps = w_opt[prob.ind_h]
    else:
        t_stages = opts.terminal_time / opts.N_stages
        for Nfe in opts.Nfe_list:
            time_steps = Nfe * [t_stages / Nfe]
    results["time_steps"] = time_steps

    # results relevant for OCP:
    x0 = prob.w0[prob.ind_x[0]]
    results["x_traj"] = [x0] + results["x_list"]
    results["u_traj"] = results["u_list"]  # duplicate name
    t_grid = np.concatenate((np.array([0.0]), np.cumsum(time_steps)))
    results["t_grid"] = t_grid
    u_grid = [0] + np.cumsum(opts.Nfe_list).tolist()
    results["t_grid_u"] = [t_grid[i] for i in u_grid]

    return results


class NosnocSolver():

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):

        # preprocess inputs
        if ocp is None:
            ocp = NosnocOcp()
        opts.preprocess()
        model.preprocess_model(opts)
        ocp.preprocess_ocp(model.x, model.u)

        if opts.initialization_strategy == InitializationStrategy.RK4_SMOOTHENED:
            model.add_smooth_step_representation()

        # store references
        self.model = model
        self.ocp = ocp
        self.opts = opts

        # create problem
        problem = NosnocProblem(opts, model, ocp)
        self.problem = problem
        self.w0 = problem.w0

        # create NLP Solver
        try:
            prob = {'f': problem.cost, 'x': problem.w, 'g': problem.g, 'p': problem.p}
            self.solver = nlpsol(model.name, 'ipopt', prob, opts.opts_casadi_nlp)
        except Exception as err:
            self.print_problem()
            print(f"{opts=}")
            print("\nerror creating solver for problem above:\n")
            print(f"\nerror is \n\n: {err}")
            import pdb
            pdb.set_trace()

    def initialize(self):
        opts = self.opts
        prob = self.problem
        ind_x0 = prob.ind_x[0]
        x0 = self.w0[ind_x0]

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_W0_START:
            for ind in prob.ind_x:
                self.w0[ind] = x0
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
            db_updated_indices = list(ind_x0)
            for i in range(opts.N_finite_elements):
                Xrk4 = rk4_on_timegrid(self.model.f_x_smooth_fun,
                                       x0=x_rk4_current,
                                       t_grid=rk4_t_grid)
                x_rk4_current = Xrk4[-1]
                # print(f"{Xrk4=}")
                for k in range(opts.n_s):
                    x_ki = Xrk4[k + 1]
                    self.w0[prob.ind_x[i + 1][k]] = x_ki
                    # NOTE: we don't use lambda_smooth_fun, since it gives negative lambdas
                    # -> infeasible. Possibly another smooth min fun could be used.
                    # However, this would be inconsistent with mu.
                    lam_ki = self.model.lambda00_fun(x_ki)
                    mu_ki = self.model.mu_smooth_fun(x_ki)
                    theta_ki = self.model.theta_smooth_fun(x_ki)
                    # print(f"{x_ki=}")
                    # print(f"theta_ki = {list(theta_ki.full())}")
                    # print(f"mu_ki = {list(mu_ki.full())}")
                    # print(f"lam_ki = {list(lam_ki.full())}\n")
                    for s in range(self.model.dims.n_sys):
                        ind_theta_s = range(sum(self.model.dims.n_f_sys[:s]),
                                            sum(self.model.dims.n_f_sys[:s + 1]))
                        self.w0[prob.ind_theta[i][k][s]] = theta_ki[ind_theta_s].full().flatten()
                        self.w0[prob.ind_lam[i][k][s]] = lam_ki[ind_theta_s].full().flatten()
                        self.w0[prob.ind_mu[i][k][s]] = mu_ki[s].full().flatten()
                        # TODO: ind_v
                    db_updated_indices += prob.ind_theta[i][k][s] + prob.ind_lam[i][k][
                        s] + prob.ind_mu[i][k][s] + prob.ind_x[i + 1][k] + prob.ind_h
                if opts.irk_time_points[-1] != 1.0:
                    raise NotImplementedError
                else:
                    # Xk_end
                    self.w0[prob.ind_x[i + 1][-1]] = x_ki
                    db_updated_indices += prob.ind_x[i + 1][-1]

            # print("w0 after RK4 init:")
            # print(self.w0)
            missing_indices = sorted(set(range(len(self.w0))) - set(db_updated_indices))
            # print(f"{missing_indices=}")
            # import pdb; pdb.set_trace()

    def solve(self) -> dict:

        self.initialize()
        opts = self.opts
        prob = self.problem
        w0 = self.w0.copy()

        w_all = [w0.copy()]
        complementarity_stats = opts.max_iter_homotopy * [None]
        cpu_time_nlp = opts.max_iter_homotopy * [None]
        nlp_iter = opts.max_iter_homotopy * [None]

        if opts.print_level:
            print('-------------------------------------------')
            print('sigma \t\t compl_res \t CPU time \t iter \t status')

        sigma_k = opts.sigma_0

        # lambda00 initialization
        x0 = w0[prob.ind_x[0]]
        lambda00 = self.model.lambda00_fun(x0).full().flatten()
        p_val = np.concatenate((np.array([opts.sigma_0]), lambda00))

        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            p_val[0] = sigma_k

            # solve NLP
            t = time.time()
            sol = self.solver(x0=w0,
                              lbg=prob.lbg,
                              ubg=prob.ubg,
                              lbx=prob.lbw,
                              ubx=prob.ubw,
                              p=p_val)
            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            solver_stats = self.solver.stats()
            status = solver_stats['return_status']
            nlp_iter[ii] = solver_stats['iter_count']
            w_opt = sol['x'].full().flatten()
            w0 = w_opt
            w_all.append(w_opt)

            complementarity_residual = prob.comp_res(w_opt, p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual

            if opts.print_level:
                print(
                    f'{sigma_k:.1e} \t {complementarity_residual:.2e} \t {cpu_time_nlp[ii]:3f} \t {nlp_iter[ii]} \t {status}'
                )
            if status not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
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

        # collect results
        results = get_results_from_primal_vector(prob, w_opt)

        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_opt

        return results

    # TODO: move this to problem?
    def set(self, field: str, value):
        prob = self.problem
        if field == 'x':
            ind_x0 = prob.ind_x[0]
            prob.w0[ind_x0] = value
            prob.lbw[ind_x0] = value
            prob.ubw[ind_x0] = value
        else:
            raise NotImplementedError()

    def print_problem(self):
        self.problem.print()
