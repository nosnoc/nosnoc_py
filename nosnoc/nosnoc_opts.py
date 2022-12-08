from dataclasses import dataclass, field

import numpy as np

from .rk_utils import generate_butcher_tableu, generate_butcher_tableu_integral
from .utils import validate
from .nosnoc_types import MpccMode, IRKSchemes, StepEquilibrationMode, CrossComplementarityMode, IrkRepresentation, PssMode, IrkRepresentation, HomotopyUpdateRule, InitializationStrategy

@dataclass
class NosnocOpts:

    # discretization
    terminal_time: float = 1.0  # TODO: make param?

    use_fesd: bool = True
    print_level: int = 0
    max_iter_homotopy: int = 0

    initialization_strategy: InitializationStrategy = InitializationStrategy.ALL_XCURRENT_W0_START

    irk_representation: IrkRepresentation = IrkRepresentation.INTEGRAL

    # IRK and FESD opts
    n_s: int = 2  # Number of IRK stages
    irk_scheme: IRKSchemes = IRKSchemes.RADAU_IIA
    cross_comp_mode: CrossComplementarityMode = CrossComplementarityMode.SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA
    mpcc_mode: MpccMode = MpccMode.SCHOLTES_INEQ

    lift_irk_differential: bool = True  # NOTE: tested in simple_sim_tests
    pss_mode: PssMode = PssMode.STEWART  # possible options: Stewart and Step
    gamma_h: float = 1.0

    # initialization - Stewart
    initial_theta: float = 1.0
    initial_lambda: float = 1.0
    initial_mu: float = 1.0
    # initialization - Step
    initial_alpha: float = 1.0  # for step only
    initial_lambda_0: float = 1.0
    initial_lambda_1: float = 1.0
    initial_beta: float = 1.0
    initial_gamma: float = 1.0

    N_finite_elements: int = 2  # of length N_stages
    Nfe_list: list = field(default_factory=list)  # of length N_stages

    # MPCC and Homotopy opts
    comp_tol: float = 1e-8
    sigma_0: float = 1.0
    sigma_N: float = 1e-8
    homotopy_update_slope: float = 0.1
    homotopy_update_exponent: float = 1.5
    homotopy_update_rule: HomotopyUpdateRule = HomotopyUpdateRule.LINEAR

    # step equilibration
    step_equilibration: StepEquilibrationMode = StepEquilibrationMode.HEURISTIC_DELTA
    step_equilibration_sigma: float = 0.1
    rho_h: float = 1.0

    # OCP only
    N_stages: int = 1
    equidistant_control_grid: bool = True  # NOTE: tested in test_ocp

    # IPOPT opts
    opts_ipopt = dict()
    opts_ipopt['print_time'] = 0
    opts_ipopt['verbose'] = False
    opts_ipopt['ipopt'] = dict()
    opts_ipopt['ipopt']['sb'] = 'yes'
    opts_ipopt['ipopt']['max_iter'] = 500
    opts_ipopt['ipopt']['print_level'] = 0
    opts_ipopt['ipopt']['bound_relax_factor'] = 0
    tol_ipopt = 1e-10
    opts_ipopt['ipopt']['tol'] = tol_ipopt
    opts_ipopt['ipopt']['dual_inf_tol'] = tol_ipopt
    opts_ipopt['ipopt']['dual_inf_tol'] = tol_ipopt
    opts_ipopt['ipopt']['compl_inf_tol'] = tol_ipopt
    opts_ipopt['ipopt']['mu_strategy'] = 'adaptive'
    opts_ipopt['ipopt']['mu_oracle'] = 'quality-function'

    time_freezing: bool = False

    def __repr__(self) -> str:
        out = ''
        for k, v in self.__dict__.items():
            out += f"{k} : {v}\n"
        return out

    def preprocess(self):
        validate(self)
        self.opts_ipopt['ipopt']['print_level'] = self.print_level

        if self.time_freezing:
            raise NotImplementedError()

        if self.max_iter_homotopy == 0:
            self.max_iter_homotopy = int(
                np.ceil(
                    np.abs(
                        np.log(self.comp_tol / self.sigma_0) /
                        np.log(self.homotopy_update_slope)))) + 1

        if len(self.Nfe_list) == 0:
            self.Nfe_list = self.N_stages * [self.N_finite_elements]

        # Butcher Tableau
        if self.irk_representation == IrkRepresentation.INTEGRAL:
            B_irk, C_irk, D_irk, irk_time_points = generate_butcher_tableu_integral(
                self.n_s, self.irk_scheme)
            self.B_irk = B_irk
            self.C_irk = C_irk
            self.D_irk = D_irk
        elif self.irk_representation == IrkRepresentation.DIFFERENTIAL:
            A_irk, b_irk, irk_time_points, _ = generate_butcher_tableu(
                self.n_s, self.irk_scheme)
            self.A_irk = A_irk
            self.b_irk = b_irk

        if np.abs(irk_time_points[-1] - 1.0) < 1e-9:
            self.right_boundary_point_explicit = True
        else:
            self.right_boundary_point_explicit = False
        return

    ## Options in matlab..
    # MPCC related, not implemented yet.
    # s_elastic_0 = 1
    # s_elastic_max = 1e1
    # s_elastic_min = 0

    # Time-Setting # NOTE: all not needed (for now)
    # Time-Freezing
    # time_freezing_inelastic: bool = False
    # time_rescaling: bool = False
    # # for time optimal problems and equidistant control grids in physical time
    # use_speed_of_time_variables: bool = True
    # local_speed_of_time_variable: bool = False
    # stagewise_clock_constraint: bool = True
    # impose_terminal_phyisical_time: bool = True
    # # S_sot_nominal = 1
    # # rho_sot = 0
    # s_sot0: float = 1.0
    # s_sot_max: float = 25.
    # s_sot_min: float = 1.0
    # time_freezing_reduced_model = 0 # analytic reduction of lifter formulation, less algebraic variables
    # time_freezing_hysteresis = 0 # do not do automatic time freezing generation for hysteresis, it is not supported yet.
    # time_freezing_nonlinear_friction_cone = 1 # 1 - use nonlienar friction cone, 0 - use polyhedral l_inf approximation.

    # time_freezing_quadrature_state = 0 # make a nonsmooth quadrature state to integrate only if physical time is running

    ## Some Nosnoc options that are not relevant here (yet)
    # n_depth_step_lifting = 2; # it is not recomended to change this (increase nonlinearity and harms convergenc), depth is number of multilinar terms to wich a lifting variables is equated to.

    # # Default opts for the barrier tuned penalty/slack variables for mpcc modes 8 do 10.
    # rho_penalty = 1e1;
    # sigma_penalty = 0;

    # ## Homotopy preprocess and polishing steps
    # h_fixed_max_iter = 1; # number of iterations that are done with fixed h in the homotopy loop
    # h_fixed_change_sigma = 1; # if this is on, do not update sigma and just solve on nlp with fixed h.
    # polishing_step = 0; # Heuristic for fixing active set, yet exerimental, not recommended to use.
    # polishing_derivative_test = 0; # check in sliding mode also the derivative of switching functions
    # h_fixed_to_free_homotopy = 0; # start with large penaly for equidistant grid, end with variable equilibrated grid.

    # ## Step equilibration
    # delta_h_regularization = 0;
    # piecewise_equidistant_grid = 0;
    # piecewise_equidistant_grid_sigma  1;
    # piecewise_equidistant_grid_slack_mode = 0;

    # # step_equilibration_penalty = 0.1;  #(rho_h in step_equilibration modde 1, as qudratic penalty)
