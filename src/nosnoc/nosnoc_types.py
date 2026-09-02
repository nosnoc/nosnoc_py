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


class FrictionModel(Enum):
    """
    Which representation of the Coulomb friction cone to use for a CLS.
    """
    CONIC = auto()
    """Exact nonlinear friction cone, ||lambda_t||_2 <= mu*lambda_n."""
    POLYHEDRAL = auto()
    """Polyhedral approximation of the friction cone spanned by D_tangent, in 2D this is equivalent to the conic model"""


class ConicModelSwitchHandling(Enum):
    """
    How switches of the tangential velocity are detected with the Conic friction model.
    """
    PLAIN = auto()
    """No extra variables, switches of the tangential velocity are not isolated."""
    ABS = auto()
    """Positive/negative parts of the tangential velocity, 0 <= p_vt perp n_vt >= 0."""
    LP = auto()
    """Positive/negative parts plus a step function alpha_vt for the tangential velocity."""


class ClsDiscretization(Enum):
    """
    Which discretization to use for the impact of a Complementarity Lagrangian System.
    """
    FESD_J = auto()
    """
    Finite Elements with Switch Detection for Jumps (default).

    The impact is exact: impulse equations at the finite element boundaries let the velocity
    *jump*, M(v^+ - v^-) = J_n Lambda_n with 0 <= Lambda_n perp Y_gap >= 0.
    """
    RELAXED_OC = auto()
    """
    Patel et al.'s relaxed orthogonal-collocation formulation (IEEE RA-L 2019).

    The velocity is continuous across finite element boundaries; the impact is produced by the
    contact force lambda_n acting over one shrinking finite element. It is an approximation at
    finite step size that converges to the exact plastic impact as h -> 0.
    """
    RELAXED_OC_IMPULSE = auto()
    """
    The relaxed OC with FESD-J's impulse block added, but not its cross complementarity.

    The finite element boundaries carry the impulse variables and Newton's restitution law,
    M(v^+ - v^-) = J_n Lambda_n with 0 <= Lambda_n perp Y_gap + P_vn + N_vn >= 0, so the velocity
    may *jump* and a nonzero coefficient of restitution is representable.

    What it does not carry is FESD-J's `lambda_normal perp Y_gap` cross complementarity, which is
    what forbids a contact force on an element whose left boundary still has a positive gap. The
    relaxed OC's *smeared* impact -- a large contact force over one collapsing element -- therefore
    remains feasible alongside the exact one: at a boundary the trajectory reaches with v^- = 0,
    because the impact already happened inside the previous element, the restitution law degenerates
    to 0 = 0 and constrains nothing.

    This mode is therefore not exact. It exists to separate the two ingredients of FESD-J: it offers
    the exact impact without forbidding the smeared one, so comparing it against `RELAXED_OC` and
    `FESD_J` tells apart which of the two rules out a spurious solution.
    """
    RELAXED_OC_IMPULSE_ONLY = auto()
    """
    Contact enters only as an impulse at the finite element boundaries, never as a force.

    Like `RELAXED_OC_IMPULSE` the boundaries carry the impulse variables and Newton's restitution
    law, but the continuous contact force `lambda_normal` does not exist at all: the right hand side
    of the ODE is pure free flight, M v' = f_v, and the whole cross complementarity
    `lambda_normal perp y_gap` is gone with it. Non-penetration comes from the lifted gap alone,
    y_gap = f_c(q) >= 0 at the RK stage points and the right boundary point.

    Because there is no force, the *smeared* impact that `RELAXED_OC` and `RELAXED_OC_IMPULSE` both
    admit -- a large contact force over one collapsing element -- is structurally impossible here
    rather than merely disfavoured by the solver. FESD-J rules it out too, but by adding a
    constraint (`lambda_normal perp Y_gap`); this mode rules it out by removing the variable.

    Note:
        Sustained contact cannot be represented. Holding a resting contact needs a steady contact
        force, and with only impulses available the discretization would have to fire one at every
        element boundary. Use it for problems whose contacts are genuinely impulsive, such as a ball
        that bounces away immediately; a body coming to rest on a surface is out of scope.
    """


class Status(Enum):
    SUCCESS = auto()
    INFEASIBLE = auto()
