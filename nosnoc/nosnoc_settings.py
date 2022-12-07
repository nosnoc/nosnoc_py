from dataclasses import dataclass, field
from enum import Enum


class MpccMode(Enum):
    SCHOLTES_INEQ = 0
    SCHOLTES_EQ = 1
    # NOSNOC: 'scholtes_ineq' (3), 'scholtes_eq' (2)
    # NOTE: tested in simple_sim_tests


class IRKSchemes(Enum):
    RADAU_IIA = 0
    GAUSS_LEGENDRE = 1
    # NOTE: tested in simple_sim_tests


class InitializationStrategy(Enum):
    ALL_XCURRENT_W0_START = 0
    OLD_SOLUTION = 1
    # Other ideas
    # RK4_ON_SMOOTHENED
    # lp_initialization


class StepEquilibrationMode(Enum):
    HEURISTIC_MEAN = 0
    HEURISTIC_DELTA = 1
    L2_RELAXED_SCALED = 2
    L2_RELAXED = 3
    # NOTE: tested in test_ocp_motor


class CrossComplementarityMode(Enum):
    COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER = 0  # nosnoc 1
    SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA = 1  # nosnoc 3
    # NOTE: tested in simple_sim_tests


class IrkRepresentation(Enum):
    INTEGRAL = 0
    DIFFERENTIAL = 1
    # NOTE: tested in test_ocp


class HomotopyUpdateRule(Enum):
    LINEAR = 0
    SUPERLINEAR = 1

class PssMode(Enum):
    # NOTE: tested in simple_sim_tests
    STEWART = 0
    """
    basic algebraic equations and complementarity condtions of the DCS
    lambda_i'*theta_i = 0; for all i = 1,..., n_simplex
    lambda_i >= 0;    for all i = 1,..., n_simplex
    theta_i >= 0;     for all i = 1,..., n_simplex
    """
    STEP = 1
    """
    c_i(x) - (lambda_p_i-lambda_n_i)  = 0; for all i = 1,..., n_simplex
    lambda_n_i'*alpha_i  = 0; for all i = 1,..., n_simplex
    lambda_p_i'*(e-alpha_i)  = 0; for all i = 1,..., n_simplex
    lambda_n_i >= 0;    for all i = 1,..., n_simplex
    lambda_p_i >= 0;    for all i = 1,..., n_simplex
    alpha_i >= 0;     for all i = 1,..., n_simplex
    """


@dataclass
class NosnocSettings:

    # discretization
    terminal_time: float = 1.0  # TODO: make param?

    use_fesd: bool = True
    print_level: int = 0
    max_iter_homotopy: int = 0

    initialization_strategy: InitializationStrategy = InitializationStrategy.ALL_XCURRENT_W0_START

    irk_representation: IrkRepresentation = IrkRepresentation.INTEGRAL

    # IRK and FESD Settings
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

    # MPCC and Homotopy Settings
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

    # IPOPT Settings
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

    # # Default settings for the barrier tuned penalty/slack variables for mpcc modes 8 do 10.
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
