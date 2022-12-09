from typing import Optional, List
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field

import numpy as np
from casadi import SX, vertcat, horzcat, sum1, inf, Function, diag, nlpsol, fabs, tanh, mmin, transpose, fmax, fmin

from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule
from nosnoc.utils import casadi_length, print_casadi_vector, casadi_vertcat_list, casadi_sum_list, flatten_layer, flatten, increment_indices


@dataclass
class NosnocModel:
    r"""
    \dot{x} \in f_i(x, u) if x(t) in R_i \subset \R^{n_x}

    with R_i = {x \in \R^{n_x} | diag(S_i,\dot) * c(x) > 0}

    where S_i denotes the rows of S.
    """
    # TODO: extend docu for n_sys > 1
    x: SX
    F: List[SX]
    c: List[SX]
    S: List[np.ndarray]
    x0: np.ndarray
    u: SX = SX.sym('u_dummy', 0, 1)
    name: str = 'nosnoc'


@dataclass
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
    lbu: np.ndarray = np.ones((0,))
    ubu: np.ndarray = np.ones((0,))
    f_q: SX = SX.zeros(1)
    f_q_T: SX = SX.zeros(1)
    g_terminal: SX = SX.zeros(0)


@dataclass
class NosnocDims:
    """
    detected automatically
    """
    nx: int = 0
    nu: int = 0
    nz: int = 0
    n_theta: int = 0
    n_sys: int = 0
    n_lift_eq: int = 0
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

    def __init__(self, dims: NosnocDims, opts: NosnocOpts, model: NosnocModel):
        super().__init__()
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
                 dims: NosnocDims,
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
        self.dims = dims
        self.opts = opts
        self.model = model

        # right boundary
        create_right_boundary_point = (opts.use_fesd and not opts.right_boundary_point_explicit and
                                       fe_idx < opts.Nfe_list[ctrl_idx] - 1)
        end_allowance = 1 if create_right_boundary_point else 0

        # Initialze index vectors. Note ind_x contains an extra element
        # in order to store the end variables
        # create_list_mat(n_s+1, 0)
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

        self.prev_fe = prev_fe

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

    def rk_stage_z(self, stage):
        idx = np.concatenate((flatten(self.ind_theta[stage]), flatten(self.ind_lam[stage]),
                              flatten(self.ind_mu[stage]), flatten(self.ind_alpha[stage]),
                              flatten(self.ind_lambda_n[stage]), flatten(self.ind_lambda_p[stage])))
        return self.w[idx]

    def Theta(self, stage=slice(None), sys=slice(None)):
        return vertcat(
            self.w[flatten(self.ind_theta[stage][sys])],
            self.w[flatten(self.ind_alpha[stage][sys])],
            np.ones(len(flatten(self.ind_alpha[stage][sys]))) -
            self.w[flatten(self.ind_alpha[stage][sys])])

    def sum_Theta(self):
        Thetas = [self.Theta(stage=ii) for ii in range(len(self.ind_theta))]
        return casadi_sum_list(Thetas)

    def h(self):
        return self.w[self.ind_h]

    def forward_simulation(self, ocp: NosnocOcp, Uk: SX):
        opts = self.opts
        model = self.model
        dims = self.dims

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
            temp = model.g_z_all_fun(self.w[self.ind_x[-1]], self.rk_stage_z(-1), Uk)
            self.add_constraint(temp[:casadi_length(temp) - dims.n_lift_eq])

        return

    def create_complementarity_constraints(self, sigma_p: SX):
        opts = self.opts
        dims = self.dims
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
        elif opts.cross_comp_mode == CrossComplementarityMode.SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA:
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

    def step_equilibration(self):
        dims = self.dims
        opts = self.opts
        if opts.use_fesd and self.fe_idx > 0:  # step equilibration only within control stages.
            delta_h_ki = self.h() - self.prev_fe.h()
            if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_MEAN:
                h_fe = opts.terminal_time / (opts.N_stages * opts.Nfe_list[self.ctrl_idx])
                self.cost += opts.rho_h * (self.h() -h_fe)**2
            elif opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA:
                self.cost += opts.rho_h * delta_h_ki**2
            elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED_SCALED:
                eta_k = self.prev_fe.sum_Lambda() * self.sum_Lambda() + \
                        self.prev_fe.sum_Theta() * self.sum_Theta()
                nu_k = 1
                for jjj in range(dims.n_theta):
                    nu_k = nu_k * eta_k[jjj]
                self.cost += opts.rho_h * tanh(nu_k / opts.step_equilibration_sigma) * delta_h_ki**2
            elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
                eta_k = self.prev_fe.sum_Lambda() * self.sum_Lambda() + \
                        self.prev_fe.sum_Theta() * self.sum_Theta()
                nu_k = 1
                for jjj in range(dims.n_theta):
                    nu_k = nu_k * eta_k[jjj]
                self.cost += opts.rho_h * nu_k * delta_h_ki**2
        return


class NosnocSolver(NosnocFormulationObject):

    def preprocess_model(self):
        # Note: checks ommitted for now.
        opts = self.opts
        dims = self.dims
        model = self.model

        dims.nx = casadi_length(self.model.x)
        dims.nu = casadi_length(self.model.u)
        dims.n_sys = len(self.model.F)

        upsilon = []

        dims.n_c_sys = [casadi_length(model.c[i]) for i in range(dims.n_sys)]
        dims.n_f_sys = [model.F[i].shape[1] for i in range(dims.n_sys)]

        g_Stewart_list = [-model.S[i] @ model.c[i] for i in range(dims.n_sys)]

        g_Stewart = casadi_vertcat_list(g_Stewart_list)
        c_all = casadi_vertcat_list(self.model.c)

        if opts.pss_mode == PssMode.STEP:
            # double the size of the vectors, since alpha, 1-alpha treated at same time
            # TODO: Is this correct? it does give an integer, not a list!
            dims.n_f_sys = np.sum(dims.n_c_sys, axis=0) * 2

        if max(dims.n_c_sys) < 2 and opts.pss_mode == PssMode.STEP:
            pss_lift_step_functions = 0
            # pss_lift_step_functions = 1; # lift the multilinear terms in the step functions;
            if opts.print_level >= 1:
                print(
                    'Info: opts.pss_lift_step_functions set to 0, as are step function selections are already entering the ODE linearly.\n'
                )

        # dimensions
        if opts.pss_mode == PssMode.STEWART:
            # NOTE: n_sys is needed decoupled systems: see FESD: "Remark on Cartesian products of Filippov systems"
            n_theta = sum(dims.n_f_sys)  # number of modes
            n_lambda = n_theta
            nz = n_theta + n_lambda + dims.n_sys
        elif opts.pss_mode == PssMode.STEP:
            n_alpha = np.sum(dims.n_c_sys)
            n_lambda_n = np.sum(dims.n_c_sys)
            n_lambda_p = np.sum(dims.n_c_sys)
            n_theta = 2 * n_alpha
            nz = n_alpha + n_lambda_n + n_lambda_p

        dims.nz = nz
        dims.n_theta = n_theta

        # create dummy finite element.
        # only use first stage
        fe = FiniteElement(dims, opts, model, ctrl_idx=0, fe_idx=0, prev_fe=None)

        if opts.pss_mode == PssMode.STEP:
            # Upsilon collects the vector for dotx = F(x)Upsilon, it is either multiaffine
            # terms or gamma from lifting
            if pss_lift_step_functions:
                raise NotImplementedError
            for ii in range(self.dims.n_sys):
                upsilon_temp = []
                S_temp = model.S[ii]
                for j in range(len(S_temp)):
                    upsilon_ij = 1
                    for k in range(len(S_temp[0, :])):
                        # create multiafine term
                        if S_temp[j, k] != 0:
                            upsilon_ij = upsilon_ij * (0.5 * (1 - S_temp[j, k]) +
                                                       S_temp[j, k] * fe.w[fe.ind_alpha[0][ii]][k])
                    upsilon_temp = vertcat(upsilon_temp, upsilon_ij)
                upsilon = horzcat(upsilon, upsilon_temp)
            # prepare for time freezing lifting and co, not implemented

        # start empty
        g_lift = SX.zeros((0, 1))
        g_switching = SX.zeros((0, 1))
        g_convex = SX.zeros((0, 1))  # equation for the convex multiplers 1 = e' \theta
        lambda00_expr = SX.zeros(0, 0)
        f_comp_residual = 0  # the orthogonality conditions diag(\theta) \lambda = 0.

        z = fe.rk_stage_z(0)
        if opts.pss_mode == PssMode.STEWART:
            # NOTE: In MATLAB, we do n_lift_eq = 1 here, but the following should be correct.
            n_lift_eq = dims.n_sys
            # n_lift_eq = 0
        elif opts.pss_mode == PssMode.STEP:
            if pss_lift_step_functions:
                raise NotImplementedError
            # eval functions for gamma and beta?
            n_lift_eq = casadi_length(g_lift)

        # Reformulate the Filippov ODE into a DCS
        f_x = SX.zeros((dims.nx, 1))

        if opts.pss_mode == PssMode.STEWART:
            for ii in range(dims.n_sys):
                f_x = f_x + model.F[ii] @ fe.w[fe.ind_theta[0][ii]]
                g_switching = vertcat(
                    g_switching,
                    g_Stewart_list[ii] - fe.w[fe.ind_lam[0][ii]] + fe.w[fe.ind_mu[0][ii]])
                g_convex = vertcat(g_convex, sum1(fe.w[fe.ind_theta[0][ii]]) - 1)
                f_comp_residual += fabs(fe.w[fe.ind_lam[0][ii]].T @ fe.w[fe.ind_theta[0][ii]])
                lambda00_expr = vertcat(lambda00_expr,
                                        g_Stewart_list[ii] - mmin(g_Stewart_list[ii]))
        elif opts.pss_mode == PssMode.STEP:
            for ii in range(dims.n_sys):
                f_x = f_x + model.F[ii] @ upsilon[:, ii]
                g_switching = vertcat(
                    g_switching,
                    model.c[ii] - fe.w[fe.ind_lambda_p[0][ii]] + fe.w[fe.ind_lambda_n[0][ii]])
                f_comp_residual += transpose(
                    fe.w[fe.ind_lambda_n[0][ii]]) @ fe.w[fe.ind_alpha[0][ii]]
                f_comp_residual += transpose(fe.w[fe.ind_lambda_p[0][ii]]) @ (
                    np.ones(dims.n_c_sys[ii]) - fe.w[fe.ind_alpha[0][ii]])
                lambda00_expr = vertcat(lambda00_expr, -fmin(model.c[ii], 0), fmax(model.c[ii], 0))

        # collect all algebraic equations
        g_z_all = vertcat(g_switching, g_convex, g_lift)  # g_lift_forces
        dims.n_lift_eq = n_lift_eq

        # CasADi functions for indicator and region constraint functions
        x = model.x
        u = model.u
        model.z = z
        model.g_Stewart_fun = Function('g_Stewart_fun', [x], [g_Stewart])
        model.c_fun = Function('c_fun', [x], [c_all])

        # dynamics
        model.f_x_fun = Function('f_x_fun', [x, z, u], [f_x])

        # lp kkt conditions without bilinear complementarity terms
        model.g_z_all_fun = Function('g_z_all_fun', [x, z, u], [g_z_all])
        model.lambda00_fun = Function('lambda00_fun', [model.x], [lambda00_expr])

        model.J_cc_fun = Function('J_cc_fun', [z], [f_comp_residual])

        # OCP
        self.ocp.g_terminal_fun = Function('g_terminal_fun', [x], [self.ocp.g_terminal])
        self.ocp.f_q_fun = Function('f_q_fun', [x, u], [self.ocp.f_q])

    def __create_control_stage(self, ctrl_idx, prev_fe):
        # Create control vars
        Uk = SX.sym(f'U_{ctrl_idx}', self.dims.nu)
        self.add_variable(Uk, self.ind_u, self.ocp.lbu, self.ocp.ubu, np.zeros((self.dims.nu,)))

        # Create Finite elements in this control stage
        control_stage = []
        for ii in range(self.opts.Nfe_list[ctrl_idx]):
            fe = FiniteElement(self.dims,
                               self.opts,
                               self.model,
                               ctrl_idx,
                               fe_idx=ii,
                               prev_fe=prev_fe)
            self._add_finite_element(fe)
            control_stage.append(fe)
            prev_fe = fe
        return control_stage

    def __create_primal_variables(self):
        # Initial
        self.fe0 = FiniteElementZero(self.dims, self.opts, self.model)

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

    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp]=None):

        super().__init__()

        if ocp is None:
            ocp = NosnocOcp()
        self.model = model
        self.ocp = ocp
        self.dims = NosnocDims()

        self.opts = opts
        opts.preprocess()
        self.preprocess_model()

        h_ctrl_stage = opts.terminal_time / opts.N_stages
        self.stages: list[list[FiniteElementBase]] = []

        # Index vectors
        self.ind_x = []
        self.ind_x_cont = []
        self.ind_v = []
        self.ind_z = []
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
                h_FE = sum([fe.h() for fe in stage])
                self.add_constraint(h_FE - h_ctrl_stage)

        # Scalar-valued complementarity residual
        if opts.use_fesd:
            J_comp = sum1(diag(fe.sum_Theta()) @ fe.sum_Lambda())
        else:
            J_comp = casadi_sum_list([
                model.J_cc_fun(fe.rk_stage_z(j))
                for j in range(opts.n_s)
                for fe in flatten(self.stages)
            ])

        # terminal constraint
        # NOTE: this was evaluated at Xk_end (expression for previous state before) which should be worse for convergence.
        g_terminal = ocp.g_terminal_fun(self.w[self.ind_x[-1][-1]])
        self.add_constraint(g_terminal)

        # Terminal numerical Time
        if opts.use_fesd:
            all_h = [fe.h() for stage in self.stages for fe in stage]
            self.add_constraint(sum(all_h) - opts.terminal_time)

        # CasADi Functions
        self.cost_fun = Function('cost_fun', [self.w], [self.cost])
        self.comp_res = Function('comp_res', [self.w, self.p], [J_comp])
        self.g_fun = Function('g_fun', [self.w, self.p], [self.g])

        # NLP Solver
        try:
            prob = {'f': self.cost, 'x': self.w, 'g': self.g, 'p': self.p}
            self.solver = nlpsol(model.name, 'ipopt', prob, opts.opts_ipopt)
        except Exception as err:
            self.print_problem()
            print(f"{opts=}")
            print("\nerror creating solver for problem above:\n")
            print(f"\nerror is \n\n: {err}")
            import pdb
            pdb.set_trace()

    def solve(self) -> dict:
        opts = self.opts
        w_all = []

        complementarity_stats = opts.max_iter_homotopy * [None]
        cpu_time_nlp = opts.max_iter_homotopy * [None]
        nlp_iter = opts.max_iter_homotopy * [None]

        if opts.print_level:
            print('-------------------------------------------')
            print('sigma \t\t compl_res \t CPU time \t iter \t status')

        w0 = self.w0
        sigma_k = opts.sigma_0

        # lambda00 initialization
        x0 = w0[self.ind_x[0]]
        lambda00 = self.model.lambda00_fun(x0).full().flatten()
        p_val = np.concatenate((np.array([opts.sigma_0]), lambda00))

        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            p_val[0] = sigma_k

            # solve NLP
            t = time.time()
            sol = self.solver(x0=w0,
                              lbg=self.lbg,
                              ubg=self.ubg,
                              lbx=self.lbw,
                              ubx=self.ubw,
                              p=p_val)
            cpu_time_nlp[ii] = time.time() - t

            # print and process solution
            solver_stats = self.solver.stats()
            status = solver_stats['return_status']
            nlp_iter[ii] = solver_stats['iter_count']
            w_opt = sol['x'].full().flatten()
            w0 = w_opt
            w_all.append(w_opt)

            complementarity_residual = self.comp_res(w_opt, p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual

            if opts.print_level:
                print(
                    f'{sigma_k:.1e} \t {complementarity_residual:.2e} \t {cpu_time_nlp[ii]:3f} \t {nlp_iter[ii]} \t {status}'
                )
            if status not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Warning: IPOPT exited with status {status}")
            if status == "Infeasible_Problem_Detected":
                print(f"WARNING: status {status}")

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
        results = dict()
        results["x_out"] = w_opt[self.ind_x[-1][-1]]
        results["x_list"] = [w_opt[ind] for ind in self.ind_x_cont]
        results["u_list"] = [w_opt[ind] for ind in self.ind_u]
        results["v_list"] = [w_opt[ind] for ind in self.ind_v]
        results["theta_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_theta]
        results["lambda_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_lam]
        results["mu_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_mu]

        # if opts.pss_mode == PssMode.STEP:
        results["alpha_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_alpha]
        results["lambda_n_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_lambda_n]
        results["lambda_p_list"] = [w_opt[flatten_layer(ind)] for ind in self.ind_lambda_p]

        if opts.use_fesd:
            time_steps = w_opt[self.ind_h]
        else:
            t_stages = opts.terminal_time / opts.N_stages
            for Nfe in opts.Nfe_list:
                time_steps = Nfe * [t_stages / Nfe]
        results["time_steps"] = time_steps
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter

        # results relevant for OCP:
        results["x_traj"] = [self.model.x0] + results["x_list"]
        results["u_traj"] = results["u_list"]  # duplicate name
        t_grid = np.concatenate((np.array([0.0]), np.cumsum(time_steps)))
        results["t_grid"] = t_grid
        u_grid = [0] + np.cumsum(opts.Nfe_list).tolist()
        results["t_grid_u"] = [t_grid[i] for i in u_grid]

        return results

    def set(self, field: str, value):
        if field == 'x':
            ind_x0 = range(self.dims.nx)
            self.w0[ind_x0] = value
            self.lbw[ind_x0] = value
            self.ubw[ind_x0] = value

            if self.opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_W0_START:
                for ind in self.ind_x:
                    self.w0[ind] = value
        else:
            raise NotImplementedError()

    def print_problem(self):
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
