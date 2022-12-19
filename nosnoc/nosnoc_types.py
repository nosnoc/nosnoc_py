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
    RK4_SMOOTHENED = 1 # experimental
    # Other ideas
    # OLD_SOLUTION = 1
    # lp_initialization


class StepEquilibrationMode(Enum):
    HEURISTIC_MEAN = 0
    HEURISTIC_DELTA = 1
    L2_RELAXED_SCALED = 2
    L2_RELAXED = 3
    # NOTE: tested in test_ocp_motor


class CrossComplementarityMode(Enum):
    COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER = 0  # nosnoc 1
    SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA = 1  # nosnoc 3
    # NOTE: tested in simple_sim_tests


class IrkRepresentation(Enum):
    INTEGRAL = 0
    DIFFERENTIAL = 1
    # NOTE: tested in test_ocp


class HomotopyUpdateRule(Enum):
    LINEAR = 0
    SUPERLINEAR = 1


class PssMode(Enum):
    # NOTE: tested in simple_sim_tests, test_ocp_motor
    STEWART = 0
    """
    basic algebraic equations and complementarity condtions of the DCS
    lambda_i'*theta_i = 0; for all i = 1,..., n_sys
    lambda_i >= 0;    for all i = 1,..., n_sys
    theta_i >= 0;     for all i = 1,..., n_sys
    """
    STEP = 1
    """
    c_i(x) - (lambda_p_i-lambda_n_i)  = 0; for all i = 1,..., n_sys
    lambda_n_i'*alpha_i  = 0; for all i = 1,..., n_sys
    lambda_p_i'*(e-alpha_i)  = 0; for all i = 1,..., n_sys
    lambda_n_i >= 0;    for all i = 1,..., n_sys
    lambda_p_i >= 0;    for all i = 1,..., n_sys
    alpha_i >= 0;     for all i = 1,..., n_sys
    """
