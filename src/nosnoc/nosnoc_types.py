from enum import Enum, auto


class RKScheme(Enum):
    RADAU_IIA = auto()
    GAUSS_LEGENDRE = auto()
    # NOTE: tested in simple_sim_tests
    def __repr__(self):
        if self == RKScheme.RADAU_IIA:
            return "Radau-IIA"
        elif self == RKScheme.GAUSS_LEGENDRE:
            return "Gauss-Legendre"

class InitializationStrategy(Enum):
    ALL_XCURRENT_W0_START = auto()
    ALL_XCURRENT_WOPT_PREV = auto()
    EXTERNAL = auto()  # let user do from outside
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
    DIRECT_HOMOTOPY = auto()
    LINEAR_COMPLEMENTARITY = auto()
    # NOTE: tested in test_ocp_motor


class CrossComplementarityMode(Enum):
    STAGE_STAGE = auto()  # nosnoc 1
    FE_STAGE = auto()  # nosnoc 3
    STAGE_FE = auto()  # nosnoc 4
    FE_FE = auto()  # nosnoc 7
    # NOTE: tested in simple_sim_tests


class RKRepresentation(Enum):
    INTEGRAL = auto()
    DIFFERENTIAL = auto()
    DIFFERENTIAL_LIFT_X = auto()
    # NOTE: tested in test_ocp


class HomotopyUpdateRule(Enum):
    LINEAR = auto()
    SUPERLINEAR = auto()


class ConstraintRelaxationMode(Enum):
    NONE = auto()
    ELL_1 = auto()
    ELL_2 = auto()
    ELL_INF = auto()


class SpeedOfTimeVariableMode(Enum):
    NONE = auto()    # No speed of time variables
    LOCAL = auto()   # Speed of time variables as control stage discontinous variables
    GLOBAL = auto()  # Single speed of time variable used across whole problem


class DcsMode(Enum):
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


class Status(Enum):
    SUCCESS = auto()
    INFEASIBLE = auto()
