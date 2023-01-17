from typing import Optional, List
from abc import ABC, abstractmethod
import time
from copy import copy
from dataclasses import dataclass, field

import numpy as np
from casadi import SX, vertcat, horzcat, sum1, inf, Function, diag, nlpsol, fabs, tanh, mmin, transpose, fmax, fmin, exp, sqrt, norm_inf

from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling
from nosnoc.utils import casadi_length, print_casadi_vector, casadi_vertcat_list, casadi_sum_list, flatten_layer, flatten, increment_indices, create_empty_list_matrix, flatten_outer_layers
from nosnoc.rk_utils import rk4_on_timegrid


class NosnocModel:
    r"""
    \dot{x} \in f_i(x, u, p_time_var, p_global, v_global) if x(t) in R_i \subset \R^{n_x}

    with R_i = {x \in \R^{n_x} | diag(S_i,\dot) * c(x) > 0}

    where S_i denotes the rows of S.


    :param x: state variables
    :param F: set of state equations for the different regions
    :param c: set of region boundaries
    :param S: determination of the boundaries region connecting
        different state equations with each boundary zone
    :param x0: initial state
    :param u: controls
    :param p_time_var: time varying parameters
    :param p_global: global parameters
    :param p_time_var_val: initial values of the time varying parameters
        (for each control stage)
    :param p_global_val: values of the global parameters
    :param v_global: additional timefree optimization variables
    :param name: name of the model
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
                 p_time_var: SX = SX.sym('p_tim_var_dummy', 0, 1),
                 p_global: SX = SX.sym('p_global_dummy', 0, 1),
                 p_time_var_val: Optional[np.ndarray] = None,
                 p_global_val: np.ndarray = np.array([]),
                 v_global: SX = SX.sym('v_global_dummy', 0, 1),
                 name: str = 'nosnoc'):
        self.x: SX = x
        self.F: List[SX] = F
        self.c: List[SX] = c
        self.S: List[np.ndarray] = S
        self.x0: np.ndarray = x0
        self.p_time_var: SX = p_time_var
        self.p_global: SX = p_global
        self.p_time_var_val: np.ndarray = p_time_var_val
        self.p_global_val: np.ndarray = p_global_val
        self.v_global = v_global
        self.u: SX = u
        self.name: str = name

        self.dims: NosnocDims = None

    def __repr__(self) -> str:
        out = ''
        for k, v in self.__dict__.items():
            out += f"{k} : {v}\n"
        return out

    def preprocess_model(self, opts: NosnocOpts):
        # detect dimensions
        n_x = casadi_length(self.x)
        n_u = casadi_length(self.u)
        n_sys = len(self.F)
        n_c_sys = [casadi_length(self.c[i]) for i in range(n_sys)]
        n_f_sys = [self.F[i].shape[1] for i in range(n_sys)]

        # sanity checks
        if not isinstance(self.F, list):
            raise ValueError("model.F should be a list.")
        if not isinstance(self.c, list):
            raise ValueError("model.c should be a list.")
        if not isinstance(self.S, list):
            raise ValueError("model.S should be a list.")

        # parameters
        n_p_glob = casadi_length(self.p_global)
        if not self.p_global_val.shape == (n_p_glob,):
            raise Exception(f"dimension of p_global_val and p_global mismatch.",
                    f"Got p_global: {self.p_global}, p_global_val {self.p_global_val}")

        n_p_time_var = casadi_length(self.p_time_var)
        if self.p_time_var_val is None:
            self.p_time_var_val = np.zeros((opts.N_stages, n_p_time_var))
        if not self.p_time_var_val.shape == (opts.N_stages, n_p_time_var):
            raise Exception(f"dimension of p_time_var_val and p_time_var mismatch.",
                    f"Got p_time_var: {self.p_time_var}, p_time_var_val {self.p_time_var_val}")
        # extend parameters for each stage
        n_p = n_p_time_var + n_p_glob
        self.p = vertcat(self.p_time_var, self.p_global)
        self.p_ctrl_stages = [SX.sym(f'p_stage{i}', n_p) for i in range(opts.N_stages)]

        self.p_val_ctrl_stages = np.zeros((opts.N_stages, n_p))
        for i in range(opts.N_stages):
            self.p_val_ctrl_stages[i, :n_p_time_var] = self.p_time_var_val[i, :]
            self.p_val_ctrl_stages[i, n_p_time_var:] = self.p_global_val

        self.dims = NosnocDims(n_x=n_x, n_u=n_u, n_sys=n_sys, n_c_sys=n_c_sys, n_f_sys=n_f_sys,
                               n_p_time_var=n_p_time_var, n_p_glob=n_p_glob)

        # g_Stewart
        g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(n_sys)]
        g_Stewart = casadi_vertcat_list(g_Stewart_list)

        # create dummy finite element - only use first stage
        dummy_ocp = NosnocOcp()
        dummy_ocp.preprocess_ocp(self)
        fe = FiniteElement(opts, self, dummy_ocp, ctrl_idx=0, fe_idx=0, prev_fe=None)

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
        f_x = SX.zeros((n_x, 1))

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
        self.g_Stewart_fun = Function('g_Stewart_fun', [self.x, self.p], [g_Stewart])
        self.c_fun = Function('c_fun', [self.x, self.p], [casadi_vertcat_list(self.c)])

        # dynamics
        self.f_x_fun = Function('f_x_fun', [self.x, z, self.u, self.p, self.v_global], [f_x])

        # lp kkt conditions without bilinear complementarity terms
        self.g_z_switching_fun = Function('g_z_switching_fun', [self.x, z, self.u, self.p], [g_switching])
        self.g_z_all_fun = Function('g_z_all_fun', [self.x, z, self.u, self.p], [g_z_all])
        self.lambda00_fun = Function('lambda00_fun', [self.x, self.p], [lambda00_expr])

        self.std_compl_res_fun = Function('std_compl_res_fun', [z, self.p], [std_compl_res])
        self.mu00_stewart_fun = Function('mu00_stewart_fun', [self.x, self.p], [mu00_stewart])

    def add_smooth_step_representation(self, smoothing_parameter: float = 1e1):
        """
        smoothing_parameter: larger -> smoother, smaller -> more exact
        """
        if smoothing_parameter <= 0:
            raise ValueError("smoothing_parameter should be > 0")

        dims = self.dims

        # smooth step function
        y = SX.sym('y')
        smooth_step_fun = Function('smooth_step_fun', [y],
                                   [(tanh(1 / smoothing_parameter * y) + 1) / 2])

        lambda_smooth = []
        g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(dims.n_sys)]

        theta_list = [SX.zeros(nf) for nf in dims.n_f_sys]
        mu_smooth_list = []
        f_x_smooth = SX.zeros((dims.n_x, 1))
        for s in range(dims.n_sys):
            n_c: int = dims.n_c_sys[s]
            alpha_expr_s = casadi_vertcat_list([smooth_step_fun(self.c[s][i]) for i in range(n_c)])

            min_in = SX.sym('min_in', dims.n_f_sys[s])
            min_out = sum1(casadi_vertcat_list([min_in[i]*exp(-1/smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))])) / \
                      sum1(casadi_vertcat_list([exp(-1/smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))]))
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
        self.theta_smooth_fun = Function('theta_smooth_fun', [self.x, self.p], [theta_smooth])
        self.mu_smooth_fun = Function('mu_smooth_fun', [self.x, self.p], [mu_smooth])
        self.lambda_smooth_fun = Function('lambda_smooth_fun', [self.x, self.p], [lambda_smooth])


class NosnocOcp:
    """
    allows to specify

    1) constraints of the form:
    lbu <= u <= ubu
    g_terminal(x_terminal) = 0

    2) cost of the form:
    f_q(x, u)  -- integrated over the time horizon
    +
    f_terminal(x_terminal) -- evaluated at the end
    """

    def __init__(self,
                 lbu: np.ndarray = np.ones((0,)),
                 ubu: np.ndarray = np.ones((0,)),
                 lbx: np.ndarray = np.ones((0,)),
                 ubx: np.ndarray = np.ones((0,)),
                 f_q: SX = SX.zeros(1),
                 f_terminal: SX = SX.zeros(1),
                 g_terminal: SX = SX.zeros(0),
                 lbv_global: np.ndarray = np.ones((0,)),
                 ubv_global: np.ndarray = np.ones((0,)),
                 v_global_guess: np.ndarray = np.ones((0,)),
                ):
        # TODO: not providing lbu, ubu should work as well!
        self.lbu: np.ndarray = lbu
        self.ubu: np.ndarray = ubu
        self.lbx: np.ndarray = lbx
        self.ubx: np.ndarray = ubx
        self.f_q: SX = f_q
        self.f_terminal: SX = f_terminal
        self.g_terminal: SX = g_terminal
        self.lbv_global: np.ndarray = lbv_global
        self.ubv_global: np.ndarray = ubv_global
        self.v_global_guess: np.ndarray = v_global_guess

    def preprocess_ocp(self, model: NosnocModel):
        dims: NosnocDims = model.dims
        self.g_terminal_fun = Function('g_terminal_fun', [model.x, model.p, model.v_global], [self.g_terminal])
        self.f_q_T_fun = Function('f_q_T_fun', [model.x, model.p, model.v_global], [self.f_terminal])
        self.f_q_fun = Function('f_q_fun', [model.x, model.u, model.p, model.v_global], [self.f_q])

        if len(self.lbx) == 0:
            self.lbx = -inf * np.ones((dims.n_x,))
        elif len(self.lbx) != dims.n_x:
            raise ValueError("lbx should be empty or of lenght n_x.")
        if len(self.ubx) == 0:
            self.ubx = inf * np.ones((dims.n_x,))
        elif len(self.ubx) != dims.n_x:
            raise ValueError("ubx should be empty or of lenght n_x.")

        # global variables
        n_v_global = casadi_length(model.v_global)
        if len(self.lbv_global) == 0:
            self.lbv_global = -inf * np.ones((n_v_global,))
        if self.lbv_global.shape != (n_v_global,):
            raise Exception("lbv_global and v_global have inconsistent shapes.")

        if len(self.ubv_global) == 0:
            self.ubv_global = -inf * np.ones((n_v_global,))
        if self.ubv_global.shape != (n_v_global,):
            raise Exception("ubv_global and v_global have inconsistent shapes.")

        if len(self.v_global_guess) == 0:
            self.v_global_guess = -inf * np.ones((n_v_global,))
        if self.v_global_guess.shape != (n_v_global,):
            raise Exception("v_global_guess and v_global have inconsistent shapes.")

@dataclass
class NosnocDims:
    """
    detected automatically
    """
    n_x: int
    n_u: int
    n_sys: int
    n_p_time_var: int
    n_p_glob: int
    n_c_sys: list
    n_f_sys: list

class NosnocFormulationObject(ABC):

    @abstractmethod
    def __init__(self):
        # optimization variables with initial guess, bounds
        self.w: SX = SX([])
        self.w0: np.array = np.array([])
        self.lbw: np.array = np.array([])
        self.ubw: np.array = np.array([])

        # constraints and bounds
        self.g: SX = SX([])
        self.lbg: np.array = np.array([])
        self.ubg: np.array = np.array([])

        # cost
        self.cost: SX = SX.zeros(1)

        # index lists
        self.ind_x: list
        self.ind_lam: list
        self.ind_lambda_n: list
        self.ind_lambda_p: list

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
        self.add_variable(SX.sym('X0', dims.n_x), self.ind_x, model.x0, model.x0, model.x0, 0)

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
        self.ind_h = []

        # create variables
        h = SX.sym(f'h_{ctrl_idx}_{fe_idx}')
        h_ctrl_stage = opts.terminal_time / opts.N_stages
        h0 = np.array([h_ctrl_stage / np.array(opts.Nfe_list[ctrl_idx])])
        ubh = (1 + opts.gamma_h) * h0
        lbh = (1 - opts.gamma_h) * h0
        self.add_step_size_variable(h, lbh, ubh, h0)

        if opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode.SCHOLTES_INEQ]:
            lb_dual = 0.0
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER:
            lb_dual = -inf

        # RK stage stuff
        for ii in range(opts.n_s):
            # state derivatives
            if (opts.irk_representation
                    in [IrkRepresentation.DIFFERENTIAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
                self.add_variable(SX.sym(f'V_{ctrl_idx}_{fe_idx}_{ii+1}', dims.n_x), self.ind_v,
                                  -inf * np.ones(dims.n_x), inf * np.ones(dims.n_x),
                                  np.zeros(dims.n_x), ii)
            # states
            if (opts.irk_representation
                    in [IrkRepresentation.INTEGRAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
                self.add_variable(SX.sym(f'X_{ctrl_idx}_{fe_idx}_{ii+1}', dims.n_x), self.ind_x,
                                  ocp.lbx, ocp.ubx, model.x0, ii)
            # algebraic variables
            if opts.pss_mode == PssMode.STEWART:
                # add thetas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'theta_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_theta, lb_dual*np.ones(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
                        opts.init_theta * np.ones(dims.n_f_sys[ij]), ii, ij)
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, lb_dual*np.ones(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
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
                        self.ind_alpha, lb_dual*np.ones(dims.n_c_sys[ij]), np.ones(dims.n_c_sys[ij]),
                        opts.init_theta * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_n
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_n_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_n,
                            lb_dual*np.ones(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), ii, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_{ii+1}_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_p, lb_dual*np.ones(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]), opts.init_mu * np.ones(dims.n_c_sys[ij]),
                        ii, ij)

        # Add right boundary points if needed
        if create_right_boundary_point:
            if opts.pss_mode == PssMode.STEWART:
                # add lambdas
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_{ctrl_idx}_{fe_idx}_end_{ij+1}', dims.n_f_sys[ij]),
                        self.ind_lam, lb_dual * np.ones(dims.n_f_sys[ij]), inf * np.ones(dims.n_f_sys[ij]),
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
                               dims.n_c_sys[ij]), self.ind_lambda_n, lb_dual*np.ones(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]),
                        opts.init_lambda * np.ones(dims.n_c_sys[ij]), opts.n_s, ij)
                # add lambda_p
                for ij in range(dims.n_sys):
                    self.add_variable(
                        SX.sym(f'lambda_p_{ctrl_idx}_{fe_idx}_end_{ij+1}',
                               dims.n_c_sys[ij]), self.ind_lambda_p, lb_dual * np.ones(dims.n_c_sys[ij]),
                        inf * np.ones(dims.n_c_sys[ij]), opts.init_mu * np.ones(dims.n_c_sys[ij]),
                        opts.n_s, ij)

        if (not opts.right_boundary_point_explicit or
                opts.irk_representation == IrkRepresentation.DIFFERENTIAL):
            # add final X variables
            self.add_variable(SX.sym(f'X_end_{ctrl_idx}_{fe_idx+1}', dims.n_x), self.ind_x,
                              ocp.lbx, ocp.ubx, model.x0, -1)

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

    def get_Theta_list(self) -> list:
        return [self.Theta(stage=ii) for ii in range(len(self.ind_theta))]

    def sum_Theta(self) -> SX:
        return casadi_sum_list(self.get_Theta_list())

    def get_Lambdas_incl_last_prev_fe(self, sys=slice(None)):
        Lambdas = [self.Lambda(stage=ii, sys=sys) for ii in range(len(self.ind_lam))]
        Lambdas += [self.prev_fe.Lambda(stage=-1, sys=sys)]
        return Lambdas

    def sum_Lambda(self, sys=slice(None)):
        """NOTE: includes the prev fes last stage lambda"""
        Lambdas = self.get_Lambdas_incl_last_prev_fe(sys)
        return casadi_sum_list(Lambdas)


    def h(self) -> SX:
        return self.w[self.ind_h]

    def forward_simulation(self, ocp: NosnocOcp, Uk: SX) -> None:
        opts = self.opts
        model = self.model

        # setup X_fe: list of x values on fe, initialize X_end
        if opts.irk_representation == IrkRepresentation.INTEGRAL:
            X_fe = [self.w[ind] for ind in self.ind_x]
            Xk_end = opts.D_irk[0] * self.prev_fe.w[self.prev_fe.ind_x[-1]]
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL:
            X_fe = []
            Xk_end = self.prev_fe.w[self.prev_fe.ind_x[-1]]
            for j in range(opts.n_s):
                x_temp = self.prev_fe.w[self.prev_fe.ind_x[-1]]
                for r in range(opts.n_s):
                    x_temp += self.h() * opts.A_irk[j, r] * self.w[self.ind_v[r]]
                X_fe.append(x_temp)
            X_fe.append(self.w[self.ind_x[-1]])
        elif opts.irk_representation == IrkRepresentation.DIFFERENTIAL_LIFT_X:
            X_fe = [self.w[ind] for ind in self.ind_x]
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



    def create_complementarity(self, x: List[SX], y: SX, sigma: SX) -> None:
        """
        adds complementarity constraints corresponding to (x_i, y) for x_i in x to the FiniteElement.

        :param x: list of SX
        :param y: SX
        :param sigma: smoothing parameter
        """
        opts = self.opts

        n = casadi_length(y)

        if opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode. SCHOLTES_INEQ]:
            # g_comp = diag(y) @ casadi_sum_list([x_i for x_i in x]) - sigma # this works too but is a bit slower.
            g_comp = diag(casadi_sum_list([x_i for x_i in x])) @ y - sigma
            # NOTE: this line should be equivalent but yield different results
            # g_comp = casadi_sum_list([diag(x_i) @ y for x_i in x]) - sigma
        elif opts.mpcc_mode == MpccMode.FISCHER_BURMEISTER:
            g_comp = SX.zeros(n, 1)
            for j in range(n):
                for x_i in x:
                    g_comp[j] += x_i[j] + y[j] - sqrt(x_i[j]**2 + y[j]**2 + sigma**2)

        n_comp = casadi_length(g_comp)
        if opts.mpcc_mode == MpccMode.SCHOLTES_INEQ:
            lb_comp = -np.inf * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))
        elif opts.mpcc_mode in [MpccMode.SCHOLTES_EQ, MpccMode.FISCHER_BURMEISTER]:
            lb_comp = 0 * np.ones((n_comp,))
            ub_comp = 0 * np.ones((n_comp,))

        self.add_constraint(g_comp, lb=lb_comp, ub=ub_comp)

        return

    def create_complementarity_constraints(self, sigma_p: SX) -> None:
        opts = self.opts
        if not opts.use_fesd:
            for j in range(opts.n_s):
                self.create_complementarity([self.Lambda(stage=j)],
                                            self.Theta(stage=j), sigma_p)
        elif opts.cross_comp_mode == CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER:
            for j in range(opts.n_s):
                # cross comp with prev_fe
                self.create_complementarity([self.Theta(stage=j)],
                            self.prev_fe.Lambda(stage=-1), sigma_p)
                for jj in range(opts.n_s):
                    # within fe
                    self.create_complementarity([self.Theta(stage=j)],
                                    self.Lambda(stage=jj), sigma_p)
        elif opts.cross_comp_mode == CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA:
            for j in range(opts.n_s):
                # Note: sum_Lambda contains last stage of prev_fe
                Lambda_list = self.get_Lambdas_incl_last_prev_fe()
                self.create_complementarity(Lambda_list, (self.Theta(stage=j)), sigma_p)
        return

    def step_equilibration(self, sigma_p: SX) -> None:
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
            self.cost += opts.rho_h * tanh(nu_k / opts.step_equilibration_sigma) * delta_h_ki**2
        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
            self.cost += opts.rho_h * nu_k * delta_h_ki**2
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT:
            self.add_constraint(nu_k*delta_h_ki)
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT_COMPLEMENTARITY:
            self.create_complementarity([nu_k], delta_h_ki, sigma_p)
            # self.add_constraint(nu_k)
        # elif opts.step_equilibration == StepEquilibrationMode.DIRECT_TANH:
        #     self.add_constraint(tanh(nu_k)*delta_h_ki)
        return


class NosnocProblem(NosnocFormulationObject):

    def __create_control_stage(self, ctrl_idx, prev_fe):
        # Create control vars
        Uk = SX.sym(f'U_{ctrl_idx}', self.model.dims.n_u)
        self.add_variable(Uk, self.ind_u, self.ocp.lbu, self.ocp.ubu,
                          np.zeros((self.model.dims.n_u,)))

        # Create Finite elements in this control stage
        control_stage = []
        for ii in range(self.opts.Nfe_list[ctrl_idx]):
            fe = FiniteElement(self.opts, self.model, self.ocp, ctrl_idx, fe_idx=ii, prev_fe=prev_fe)
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
        self.p = vertcat(self.p, lambda00)
        self.p = vertcat(self.p, x0)

        # v_global
        self.add_variable(self.model.v_global, self.ind_v_global, self.ocp.lbv_global, self.ocp.ubv_global,
                          self.ocp.v_global_guess)

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
        self.opts = opts
        if ocp is None:
            self.ocp_trivial = True
            ocp = NosnocOcp()
        else:
            self.ocp_trivial = False
        ocp.preprocess_ocp(model)
        self.ocp = ocp

        h_ctrl_stage = opts.terminal_time / opts.N_stages
        self.stages: list[list[FiniteElementBase]] = []

        # Index vectors
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

        # setup parameters, lambda00 is added later:
        sigma_p = SX.sym('sigma_p')  # homotopy parameter
        self.p = vertcat(casadi_vertcat_list(model.p_ctrl_stages), sigma_p)

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
                fe.step_equilibration(sigma_p)

                # 4) add cost and constraints from FE to problem
                self.cost += fe.cost
                self.add_constraint(fe.g, fe.lbg, fe.ubg)

            if opts.use_fesd and opts.equidistant_control_grid:
                self.add_constraint(sum([fe.h() for fe in stage]) - h_ctrl_stage)

        # Scalar-valued complementarity residual
        if opts.use_fesd:
            J_comp = 0.0
            for fe in flatten(self.stages):
                sum_abs_lam = casadi_sum_list([fabs(lam) for lam in fe.get_Lambdas_incl_last_prev_fe()])
                sum_abs_theta = casadi_sum_list([fabs(t) for t in fe.get_Theta_list()])
                J_comp += sum1(diag(sum_abs_theta) @ sum_abs_lam)
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
        self.cost_fun = Function('cost_fun', [self.w], [self.cost])
        self.comp_res = Function('comp_res', [self.w, self.p], [J_comp])
        self.g_fun = Function('g_fun', [self.w, self.p], [self.g])

        # copy original w0
        self.w0_original = copy(self.w0)

        # LEAST_SQUARES reformulation
        if opts.constraint_handling == ConstraintHandling.LEAST_SQUARES:
            self.g_lsq = copy(self.g)
            for ii in range(casadi_length(self.g)):
                if self.lbg[ii] != 0.0:
                    raise Exception(f"least_squares constraint handling only supported if all lbg, ubg == 0.0, got {self.lbg[ii]=}, {self.ubg[ii]=}, {self.g[ii]=}")
                self.cost += self.g[ii] ** 2
            self.g = SX([])
            self.lbg = np.array([])
            self.ubg = np.array([])


    def print(self):
        print("g:")
        print_casadi_vector(self.g)
        print(f"lbg, ubg\n{np.vstack((self.lbg, self.ubg)).T}")
        print("\nw \t\t\t w0 \t\t lbw \t\t ubw:")
        for i in range(len(self.lbw)):
            print(f"{self.w[i].name():<15} \t {self.w0[i]:4f} \t {self.lbw[i]:3f} \t {self.ubw[i]:3f}")
        print(f"\ncost:\n{self.cost}")


    def is_sim_problem(self):
        if self.model.dims.n_u != 0:
            return False
        if self.opts.N_stages != 1:
            return False
        if not self.ocp_trivial:
            return False
        return True

def get_cont_algebraic_indices(ind_alg: list):
    return [ind_rk[-1] for ind_fe in ind_alg for ind_rk in ind_fe]

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
    results["alpha_list"] = [w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_alpha)]
    results["lambda_n_list"] = [w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_lambda_n)]
    results["lambda_p_list"] = [w_opt[flatten_layer(ind)] for ind in get_cont_algebraic_indices(prob.ind_lambda_p)]

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

    return results


class NosnocSolver():
    """ Main solver class which generates and solves an NLP based on the given options, dynamic model, and (optionally) the ocp data.
    """
    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        """Constructor.
        """

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
        self.problem = problem

        # create NLP Solver
        try:
            prob = {'f': problem.cost, 'x': problem.w, 'g': problem.g, 'p': problem.p}
            self.solver = nlpsol(model.name, 'ipopt', prob, opts.opts_casadi_nlp)
        except Exception as err:
            self.print_problem()
            print(f"{opts=}")
            print("\nerror creating solver for problem above:\n")
            print(f"\nerror is \n\n: {err}")
            breakpoint()

    def initialize(self):
        opts = self.opts
        prob = self.problem
        x0 = prob.model.x0

        if opts.initialization_strategy in [InitializationStrategy.ALL_XCURRENT_W0_START, InitializationStrategy.ALL_XCURRENT_WOPT_PREV]:
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

    def solve(self) -> dict:
        """ Solves the NLP with the currently stored parameters.

        :return: Returns a dictionary containing ... TODO document all fields
        """
        self.initialize()
        opts = self.opts
        prob = self.problem
        w0 = prob.w0

        w_all = [w0.copy()]
        complementarity_stats = opts.max_iter_homotopy * [None]
        cpu_time_nlp = opts.max_iter_homotopy * [None]
        nlp_iter = opts.max_iter_homotopy * [None]

        if opts.print_level:
            print('-------------------------------------------')
            print('sigma \t\t compl_res \t nlp_res \t cost_val \t CPU time \t iter \t status')

        sigma_k = opts.sigma_0

        # lambda00 initialization
        x0 = prob.model.x0
        p0 = prob.model.p_val_ctrl_stages[0]
        lambda00 = self.model.lambda00_fun(x0, p0).full().flatten()

        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            p_val = np.concatenate((prob.model.p_val_ctrl_stages.flatten(), np.array([sigma_k]), lambda00, x0))

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
            nlp_res = norm_inf(sol['g']).full()[0][0]
            cost_val = norm_inf(sol['f']).full()[0][0]
            w_opt = sol['x'].full().flatten()
            w0 = w_opt
            w_all.append(w_opt)

            complementarity_residual = prob.comp_res(w_opt, p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual

            if opts.print_level:
                print(
                    f'{sigma_k:.1e} \t {complementarity_residual:.2e} \t {nlp_res:.2e} \t {cost_val:.2e} \t {cpu_time_nlp[ii]:3f} \t {nlp_iter[ii]} \t {status}'
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

        # print constraint violation
        if opts.print_level > 1 and opts.constraint_handling == ConstraintHandling.LEAST_SQUARES:
            threshold = np.max([np.sqrt(cost_val) / 10, opts.comp_tol])
            g_val = prob.g_fun(w_opt, p_val).full().flatten()
            if max(abs(g_val)) > threshold:
                print("\nconstraint violations:")
                for ii in range(len(g_val)):
                    if g_val[ii] > threshold:
                        print(f"g_val[{ii}] = {g_val[ii]} expr: {prob.g_lsq[ii]}")

        # if cost_val > opts.comp_tol * 1e2:
        #     breakpoint()

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_WOPT_PREV:
            prob.w0[:] = w_opt[:]
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_opt

        # for i in range(len(w_opt)):
        #     print(f"w{i}: {prob.w[i]} = {w_opt[i]}")
        return results

    # TODO: move this to problem?
    def set(self, field: str, value: np.ndarray) -> None:
        """
        :param field: in ["x", "p_global", "p_time_var", "w"]
        :param value: np.ndarray: numerical value of appropriate size
        """
        prob = self.problem
        dims = prob.model.dims
        if field == 'x':
            prob.model.x0 = value
        elif field == 'p_global':
            for i in range(self.opts.N_stages):
                self.model.p_val_ctrl_stages[i, dims.n_p_time_var:] = value
        elif field == 'p_time_var':
            for i in range(self.opts.N_stages):
                self.model.p_val_ctrl_stages[i, :dims.n_p_time_var] = value[i, :]
        elif field == 'w':
            prob.w0 = value
            if self.opts.initialization_strategy is not InitializationStrategy.EXTERNAL:
                raise Warning('full initialization w might be overwritten due to InitializationStrategy != EXTERNAL.')
        else:
            raise NotImplementedError()

    def print_problem(self):
        self.problem.print()
