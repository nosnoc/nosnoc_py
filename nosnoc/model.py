from typing import Optional, List

import numpy as np
import casadi as ca

from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.dims import NosnocDims
from nosnoc.nosnoc_types import PssMode
from nosnoc.utils import casadi_length, casadi_vertcat_list


class NosnocModel:
    r"""
    \dot{x} \in f_i(x, u, p_time_var, p_global, v_global) if x(t) in R_i \subset \R^{n_x}

    with R_i = {x \in \R^{n_x} | diag(S_i,\dot) * c(x) > 0}

    where S_i denotes the rows of S.

    An alternate model can be used if your system cannot be defined as a Fillipov system.
    in that case user can provide an \alpha vector (analogous to the \alpha defined in
    the step reformulation) with the same size as c, along with the expression for \dot{x}.
    This alpha vector is then used as in the Step reformulation to do switch detection.

    \dot{x} = f_x(x,u,\alpha,p)

    :param x: state variables
    :param x0: initial state
    :param F: set of state equations for the different regions
    :param c: set of region boundaries
    :param S: determination of the boundaries region connecting
        different state equations with each boundary zone
    :param g_Stewart: List of stewart functions to define the regions (instead of S & c)
    :param u: controls
    :param alpha: optionally provided alpha variables for general inclusions
    :param f_x: optionally provided rhs used for general inclusions
    :param p_time_var: time varying parameters
    :param p_global: global parameters
    :param p_time_var_val: initial values of the time varying parameters
        (for each control stage)
    :param p_global_val: values of the global parameters
    :param v_global: additional timefree optimization variables
    :param t_var: time variable (for time freezing)
    :param name: name of the model
    """

    # TODO: extend docu for n_sys > 1
    # NOTE: n_sys is needed decoupled systems: see FESD: "Remark on Cartesian products of Filippov systems"
    def __init__(self,
                 x: ca.SX,
                 x0: Optional[np.ndarray],
                 F: Optional[List[ca.SX]] = None,
                 c: Optional[List[ca.SX]] = None,
                 S: Optional[List[np.ndarray]] = None,
                 g_Stewart: Optional[List[ca.SX]] = None,
                 u: ca.SX = ca.SX.sym('u_dummy', 0, 1),
                 alpha: List[ca.SX] = [],
                 f_x: Optional[List[ca.SX]] = None,
                 p_time_var: ca.SX = ca.SX.sym('p_tim_var_dummy', 0, 1),
                 p_global: ca.SX = ca.SX.sym('p_global_dummy', 0, 1),
                 p_time_var_val: Optional[np.ndarray] = None,
                 p_global_val: np.ndarray = np.array([]),
                 v_global: ca.SX = ca.SX.sym('v_global_dummy', 0, 1),
                 t_var: Optional[ca.SX] = None,
                 name: str = 'nosnoc'):
        self.x: ca.SX = x
        self.alpha: ca.SX = alpha
        self.F: Optional[List[ca.SX]] = F
        self.f_x: List[ca.SX] = f_x
        self.c: List[ca.SX] = c
        self.S: List[np.ndarray] = S
        self.g_Stewart = g_Stewart

        if not (bool(F is not None) ^ bool((f_x is not None) and (casadi_length(casadi_vertcat_list(alpha)) != 0))):
            raise ValueError("Provide either F (Fillipov) or f_x and alpha")
        # Either c and S or g is given!
        if F is not None:
            if not (bool(c is not None and S is not None) ^ bool(g_Stewart is not None)):
                raise ValueError("Provide either c and S or g, not both!")

        self.x0: np.ndarray = x0
        self.p_time_var: ca.SX = p_time_var
        self.p_global: ca.SX = p_global
        self.p_time_var_val: np.ndarray = p_time_var_val
        self.p_global_val: np.ndarray = p_global_val
        self.v_global = v_global
        self.u: ca.SX = u
        self.t_var: ca.SX = t_var
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
        n_sys = len(self.F) if self.F is not None else len(self.f_x)
        if self.g_Stewart:
            n_c_sys = [0]  # No c used!
        else:
            n_c_sys = [casadi_length(self.c[i]) for i in range(n_sys)]

        # sanity checks
        if self.F is not None:
            n_f_sys = [self.F[i].shape[1] for i in range(n_sys)]
            if not isinstance(self.F, list):
                raise ValueError("model.F should be a list.")
            for i, f in enumerate(self.F):
                if f.shape[1] == 1:
                    raise Warning(f"model.F item {i} is not a switching system!")

            if self.g_Stewart:
                if opts.pss_mode != PssMode.STEWART:
                    raise ValueError("model.g_Stewart is used and pss_mode should be STEWART")
                if not isinstance(self.g_Stewart, list):
                    raise ValueError("model.g_Stewart should be a list.")
                for i, g_Stewart in enumerate(self.g_Stewart):
                    if g_Stewart.shape[0] != self.F[i].shape[1]:
                        raise ValueError(f"Dimensions g_Stewart and F for item {i} should be equal!")
            else:
                if not isinstance(self.c, list):
                    raise ValueError("model.c should be a list.")
                if not isinstance(self.S, list):
                    raise ValueError("model.S should be a list.")
                if len(self.c) != len(self.S):
                    raise ValueError("model.c and model.S should have the same length!")
                for i, (fi, Si) in enumerate(zip(self.F, self.S)):
                    if fi.shape[1] != Si.shape[0]:
                        raise ValueError(
                            f"model.F item {i} and S {i} should have the same number of columns")
        else:  # Only check Step because stewart is not allowed for general inclusions
            n_f_sys = [self.f_x[i].shape[1] for i in range(n_sys)]
            if not isinstance(self.c, list):
                raise ValueError("model.c should be a list.")
            if casadi_length(casadi_vertcat_list(self.c)) != casadi_length(casadi_vertcat_list(self.alpha)):
                raise ValueError("model.c and model.alpha should have the same length!")

        # parameters
        n_p_glob = casadi_length(self.p_global)
        if not self.p_global_val.shape == (n_p_glob,):
            raise Exception("dimension of p_global_val and p_global mismatch.",
                            f"Expected shape ({n_p_glob},), got p_global_val.shape {self.p_global_val.shape},"
                            f"p_global {self.p_global}, p_global_val {self.p_global_val}")

        n_p_time_var = casadi_length(self.p_time_var)
        if self.p_time_var_val is None:
            self.p_time_var_val = np.zeros((opts.N_stages, n_p_time_var))
        if not self.p_time_var_val.shape == (opts.N_stages, n_p_time_var):
            raise Exception(
                "dimension of p_time_var_val and p_time_var mismatch.",
                f"Expected shape: ({opts.N_stages}, {n_p_time_var}), "
                f"got p_time_var_val.shape {self.p_time_var_val.shape}"
                f"p_time_var {self.p_time_var}, p_time_var_val {self.p_time_var_val}")
        # extend parameters for each stage
        n_p = n_p_time_var + n_p_glob
        self.p = ca.vertcat(self.p_time_var, self.p_global)
        self.p_ctrl_stages = [ca.SX.sym(f'p_stage{i}', n_p) for i in range(opts.N_stages)]

        self.p_val_ctrl_stages = np.zeros((opts.N_stages, n_p))
        for i in range(opts.N_stages):
            self.p_val_ctrl_stages[i, :n_p_time_var] = self.p_time_var_val[i, :]
            self.p_val_ctrl_stages[i, n_p_time_var:] = self.p_global_val

        self.dims = NosnocDims(n_x=n_x,
                               n_u=n_u,
                               n_sys=n_sys,
                               n_c_sys=n_c_sys,
                               n_f_sys=n_f_sys,
                               n_p_time_var=n_p_time_var,
                               n_p_glob=n_p_glob)

        if opts.pss_mode == PssMode.STEWART:
            if self.g_Stewart:
                g_Stewart_list = self.g_Stewart
            else:
                g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(n_sys)]

            g_Stewart = casadi_vertcat_list(g_Stewart_list)

        # create dummy finite element - only use first stage
        if len(self.alpha) != 0:
            theta, lam, mu, _, lambda_n, lambda_p = self.create_stage_vars(opts)
            alpha = self.alpha
        else:
            theta, lam, mu, alpha, lambda_n, lambda_p = self.create_stage_vars(opts)

        # setup upsilon
        upsilon = []
        if opts.pss_mode == PssMode.STEP and self.F is not None:
            for ii in range(self.dims.n_sys):
                upsilon_temp = []
                S_temp = self.S[ii]
                for j in range(len(S_temp)):
                    upsilon_ij = 1
                    for k in range(len(S_temp[0, :])):
                        # create multiafine term
                        if S_temp[j, k] != 0:
                            upsilon_ij *= (0.5 * (1 - S_temp[j, k]) +
                                           S_temp[j, k] * alpha[ii][k])
                    upsilon_temp = ca.vertcat(upsilon_temp, upsilon_ij)
                upsilon = ca.horzcat(upsilon, upsilon_temp)

        # start empty
        g_lift = ca.SX.zeros((0, 1))
        g_switching = ca.SX.zeros((0, 1))
        g_convex = ca.SX.zeros((0, 1))  # equation for the convex multiplers 1 = e' \theta
        lambda00_expr = ca.SX.zeros(0, 0)
        std_compl_res = ca.SX.zeros(1)  # residual of standard complementarity

        z = ca.vertcat(casadi_vertcat_list(theta),
                       casadi_vertcat_list(lam),
                       casadi_vertcat_list(mu),
                       casadi_vertcat_list(alpha),
                       casadi_vertcat_list(lambda_n),
                       casadi_vertcat_list(lambda_p))

        # Reformulate the Filippov ODE into a DCS
        if self.F is None:
            f_x = casadi_vertcat_list(self.f_x)
            z[0:sum(n_c_sys)] = casadi_vertcat_list(self.alpha)
        else:
            f_x = ca.SX.zeros((n_x, 1))

        if opts.pss_mode == PssMode.STEWART:
            for ii in range(n_sys):
                f_x = f_x + self.F[ii] @ theta[ii]
                g_switching = ca.vertcat(
                    g_switching,
                    g_Stewart_list[ii] - lam[ii] + mu[ii])
                g_convex = ca.vertcat(g_convex, ca.sum1(theta[ii]) - 1)
                std_compl_res += ca.fabs(lam[ii].T @ theta[ii])
                lambda00_expr = ca.vertcat(lambda00_expr,
                                           g_Stewart_list[ii] - ca.mmin(g_Stewart_list[ii]))
        elif opts.pss_mode == PssMode.STEP:
            for ii in range(n_sys):
                if self.F is not None:
                    f_x = f_x + self.F[ii] @ upsilon[:, ii]
                    alpha_ii = alpha[ii]
                else:
                    alpha_ii = self.alpha[ii]
                g_switching = ca.vertcat(
                    g_switching,
                    self.c[ii] - lambda_p[ii] + lambda_n[ii])
                std_compl_res += ca.transpose(lambda_n[ii]) @ alpha_ii
                std_compl_res += ca.transpose(lambda_p[ii]) @ (np.ones(n_c_sys[ii]) - alpha_ii)
                lambda00_expr = ca.vertcat(lambda00_expr, -ca.fmin(self.c[ii], 0),
                                           ca.fmax(self.c[ii], 0))

        # collect all algebraic equations
        g_z_all = ca.vertcat(g_switching, g_convex, g_lift)  # g_lift_forces

        # CasADi functions for indicator and region constraint functions
        self.z = z

        # dynamics
        self.f_x_fun = ca.Function('f_x_fun', [self.x, z, self.u, self.p, self.v_global], [f_x])

        # lp kkt conditions without bilinear complementarity terms
        self.g_z_switching_fun = ca.Function('g_z_switching_fun', [self.x, z, self.u, self.p],
                                             [g_switching])
        self.g_z_all_fun = ca.Function('g_z_all_fun', [self.x, z, self.u, self.p], [g_z_all])
        if self.t_var is not None:
            self.t_fun = ca.Function("t_fun", [self.x], [self.t_var])
        elif opts.time_freezing:
            raise ValueError("Please provide t_var for time freezing!")

        self.lambda00_fun = ca.Function('lambda00_fun', [self.x, self.p], [lambda00_expr])

        self.std_compl_res_fun = ca.Function('std_compl_res_fun', [z, self.p], [std_compl_res])
        if opts.pss_mode == PssMode.STEWART:
            mu00_stewart = casadi_vertcat_list([ca.mmin(g_Stewart_list[ii]) for ii in range(n_sys)])
            self.mu00_stewart_fun = ca.Function('mu00_stewart_fun', [self.x, self.p], [mu00_stewart])
            self.g_Stewart_fun = ca.Function('g_Stewart_fun', [self.x, self.p], [g_Stewart])

    def create_stage_vars(self, opts):
        """
        Create the algebraic vars for a single stage.
        """
        # algebraic variables
        if opts.pss_mode == PssMode.STEWART:
            # add thetas
            theta = [ca.SX.sym('theta', self.dims.n_f_sys[ij]) for ij in range(self.dims.n_sys)]
            # add lambdas
            lam = [ca.SX.sym('lambda', self.dims.n_f_sys[ij]) for ij in range(self.dims.n_sys)]
            # add mu
            mu = [ca.SX.sym('mu', 1) for ij in range(self.dims.n_sys)]

            # unused
            alpha = []
            lambda_n = []
            lambda_p = []
        elif opts.pss_mode == PssMode.STEP:
            # add alpha
            alpha = [ca.SX.sym('alpha', self.dims.n_c_sys[ij]) for ij in range(self.dims.n_sys)]
            # add lambda_n
            lambda_n = [ca.SX.sym('lambda_n', self.dims.n_c_sys[ij]) for ij in range(self.dims.n_sys)]
            # add lambda_p
            lambda_p = [ca.SX.sym('lambda_p', self.dims.n_c_sys[ij]) for ij in range(self.dims.n_sys)]

            # unused
            theta = []
            lam = []
            mu = []
        return theta, lam, mu, alpha, lambda_n, lambda_p

    def add_smooth_step_representation(self, smoothing_parameter: float = 1e1):
        """
        smoothing_parameter: larger -> smoother, smaller -> more exact
        """
        if self.g_Stewart:
            raise NotImplementedError()
        if smoothing_parameter <= 0:
            raise ValueError("smoothing_parameter should be > 0")

        dims = self.dims

        # smooth step function
        y = ca.SX.sym('y')
        smooth_step_fun = ca.Function('smooth_step_fun', [y],
                                      [(ca.tanh(1 / smoothing_parameter * y) + 1) / 2])

        lambda_smooth = []
        g_Stewart_list = [-self.S[i] @ self.c[i] for i in range(dims.n_sys)]

        theta_list = [ca.SX.zeros(nf) for nf in dims.n_f_sys]
        mu_smooth_list = []
        f_x_smooth = ca.SX.zeros((dims.n_x, 1))
        for s in range(dims.n_sys):
            n_c: int = dims.n_c_sys[s]
            alpha_expr_s = casadi_vertcat_list([smooth_step_fun(self.c[s][i]) for i in range(n_c)])

            min_in = ca.SX.sym('min_in', dims.n_f_sys[s])
            min_out = ca.sum1(casadi_vertcat_list([min_in[i]*ca.exp(-1/smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))])) / \
                      ca.sum1(casadi_vertcat_list([ca.exp(-1/smoothing_parameter * min_in[i]) for i in range(casadi_length(min_in))]))
            smooth_min_fun = ca.Function('smooth_min_fun', [min_in], [min_out])
            mu_smooth_list.append(-smooth_min_fun(g_Stewart_list[s]))
            lambda_smooth = ca.vertcat(lambda_smooth,
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

        self.f_x_smooth_fun = ca.Function('f_x_smooth_fun', [self.x], [f_x_smooth])
        self.theta_smooth_fun = ca.Function('theta_smooth_fun', [self.x, self.p], [theta_smooth])
        self.mu_smooth_fun = ca.Function('mu_smooth_fun', [self.x, self.p], [mu_smooth])
        self.lambda_smooth_fun = ca.Function('lambda_smooth_fun', [self.x, self.p], [lambda_smooth])
