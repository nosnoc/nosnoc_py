from enum import Enum, auto


class MpccMode(Enum):
    SCHOLTES_INEQ = auto()
    SCHOLTES_EQ = auto()
    FISCHER_BURMEISTER = auto()
    # KANZOW_SCHWARTZ = auto()
    # NOSNOC: 'scholtes_ineq' (3), 'scholtes_eq' (2)
    # NOTE: tested in simple_sim_tests


class IrkSchemes(Enum):
    RADAU_IIA = auto()
    GAUSS_LEGENDRE = auto()
    # NOTE: tested in simple_sim_tests


class InitializationStrategy(Enum):
    ALL_XCURRENT_W0_START = auto()
    ALL_XCURRENT_WOPT_PREV = auto()
    EXTERNAL = auto() # let user do from outsid
    RK4_SMOOTHENED = auto()  # experimental
    # Other ideas
    # OLD_SOLUTION = auto()
    # lp_initialization


class StepEquilibrationMode(Enum):
    HEURISTIC_MEAN = auto()
    HEURISTIC_DELTA = auto()
    L2_RELAXED_SCALED = auto()
    L2_RELAXED = auto()
    DIRECT = auto()
    DIRECT_COMPLEMENTARITY = auto()
    # NOTE: tested in test_ocp_motor


class CrossComplementarityMode(Enum):
    COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER = auto()  # nosnoc 1
    SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA = auto()  # nosnoc 3
    # NOTE: tested in simple_sim_tests


class IrkRepresentation(Enum):
    INTEGRAL = auto()
    DIFFERENTIAL = auto()
    DIFFERENTIAL_LIFT_X = auto()
    # NOTE: tested in test_ocp


class HomotopyUpdateRule(Enum):
    LINEAR = auto()
    SUPERLINEAR = auto()


class ConstraintHandling(Enum):
    EXACT = auto()
    LEAST_SQUARES = auto()

class PssMode(Enum):
    """
    Mode to represent the Piecewise Smooth System (PSS).
    """
    # NOTE: tested in simple_sim_tests, test_ocp_motor
    STEWART = auto()
    """
    Stewart representaion

    basic algebraic equations and complementarity condtions of the DCS
    lambda_i'*theta_i = 0; for all i = 1,..., n_sys
    lambda_i >= 0;    for all i = 1,..., n_sys
    theta_i >= 0;     for all i = 1,..., n_sys
    """
    STEP = auto()
    """
    Step representaion

    c_i(x) - (lambda_p_i-lambda_n_i)  = 0; for all i = 1,..., n_sys
    lambda_n_i'*alpha_i  = 0; for all i = 1,..., n_sys
    lambda_p_i'*(e-alpha_i)  = 0; for all i = 1,..., n_sys
    lambda_n_i >= 0;    for all i = 1,..., n_sys
    lambda_p_i >= 0;    for all i = 1,..., n_sys
    alpha_i >= 0;     for all i = 1,..., n_sys
    """
