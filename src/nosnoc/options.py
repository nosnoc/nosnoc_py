from typing import Type, List, Optional
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .nosnoc_types import RKScheme, StepEquilibrationMode, CrossComplementarityMode, RKRepresentation, DcsMode, HomotopyUpdateRule, InitializationStrategy, SpeedOfTimeVariableMode, ConstraintRelaxationMode

@dataclass
class Options():
    h: Optional[float] = None
    h_k: Optional[List[float]] = None
    T: Optional[float] = None

    # boolean: If true the FESD discretization is used, otherwise a direct time-stepping discretization is used.
    use_fesd: bool = True

    # string: Which casadi symbolics to use. Can either be `'casadi.SX'` or `'casadi.MX'.`
    casadi_symbolic_mode: Type = ca.SX

    N_stages: int = 1 # int: Number of control stages.

    # int: Number of finite elements in each control stage. This can either be a scalar value
    # in which case it is transformed into a vector for that value when :meth:`preprocess` is called.
    # Alternatively you can pass a vector of size :attr:`N_stages`.
    N_finite_elements: int|List[int] = 2

    n_s: int = 2 # int: Number of Stages in the Runge-Kutta scheme.

    # RKSchemes: Which Runge-Kutta scheme family to use.
    #
    # See Also:
    #    `RKSchemes` for more details as to the how to choose a Runge-Kutta Scheme and
    #    for differences between them.
    rk_scheme: RKScheme = RKScheme.RADAU_IIA

    # RKRepresentation: Which representation of Runge-Kutta discretization to use.
    #
    # See Also:
    #     `RKRepresentation` for a description of the representations.
    rk_representation: RKRepresentation = RKRepresentation.INTEGRAL

    # CrossCompMode: Which cross complementarity mode to use.
    #
    # See Also:
    #     `CrossCompMode` for a description of the representations.
    cross_comp_mode: CrossComplementarityMode = CrossComplementarityMode.FE_FE

    # double: Fraction in the range $\gamma_h \in [0,1]$ by which the step size is relaxed:
    # $$(1-\gamma_h) h_0\le h \le (1+\gamma_h) h_0$$
    gamma_h: float  = 1

    dcs_mode: DcsMode = DcsMode.STEWART # DcsMode: Which DCS to reformulate the problem into.

    #--------------------- Initial Values ---------------------#

    initial_alpha: float      = 0.5 # double: Initial value for $\alpha$ in the Heaviside step reformulation.
    initial_lambda_n: float   = 0.5 # double: Initial value for $\lambda_n$ in the Heaviside step reformulation.
    initial_lambda_p: float   = 0.5 # double: Initial value for $\lambda_p$ in the Heaviside step reformulation.
    initial_beta_lift: float  = 1 # double: Initial value for $\beta$ when lifting is enabled in the Heaviside step reformulation.
    initial_theta_step: float = 1 # double: Initial value for $\theta$ when lifting is enabled in the Heaviside step reformulation.
    initial_lambda_gcs: float = 0 # double: Initial value for $\lambda$ in the Gradient Comlementarity System.

    initial_Lambda_normal: float  = 0 # double: Initial value for $\Lambda_n$ in FESD-J reformulation.
    initial_P_vn: float           = 1 # double: Initial value for positive normal velocity slack in FESD-J reformulation impulse calculation.
    initial_N_vn: float           = 1 # double: Initial value for negative normal velocity slack in FESD-J reformulation impulse calculation.
    initial_Y_gap: float          = 1 # double: Initial value for gap function in FESD-J reformulation impulse calculation.
    initial_Lambda_tangent: float = 1 # double: Initial value for $\Lambda_t$ in FESD-J reformulation impulse calculation.
    initial_Gamma_d: float        = 1 # double: Initial value for $\Gamma_d$ in FESD-J reformulation impulse calculation.
    initial_Beta_d: float         = 1 # double: Initial value for $\Beta_d$ in FESD-J reformulation impulse calculation.
    initial_Delta_d: float        = 1 # double: Initial value for $\Delta_d$ in FESD-J reformulation impulse calculation.
    initial_Gamma: float          = 1 # double: Initial value for $\Gamma$ in FESD-J reformulation impulse calculation.
    initial_Beta: float           = 1 # double: Initial value for $\Beta$ in FESD-J reformulation impulse calculation.
    initial_P_vt: float           = 1 # double: Initial value for positive tangential velocity slack in FESD-J reformulation impulse calculation.
    initial_N_vt: float           = 1 # double: Initial value for negative tangential velocity slack in FESD-J reformulation impulse calculation.
    initial_Alpha_vt: float       = 1 # double: Initial value fo tangential velocity step function in FESD-J reformulation impulse calculation.

    initial_lambda_normal: float  = 1 # double: Initial value for $\lambda_n$ in FESD-J reformulation.
    initial_p_vn: float           = 1 # double: Initial value for positive normal velocity slack in FESD-J reformulation.
    initial_n_vn: float           = 1 # double: Initial value for negative normal velocity slack in FESD-J reformulation.
    initial_y_gap: float          = 1 # double: Initial value for gap function in FESD-J reformulation.
    initial_lambda_tangent: float = 1 # double: Initial value for $\lambda_t$ in FESD-J reformulation.
    initial_gamma_d: float        = 1 # double: Initial value for $\gamma_d$ in FESD-J reformulation.
    initial_beta_d: float         = 1 # double: Initial value for $\beta_d$ in FESD-J reformulation.
    initial_delta_d: float        = 1 # double: Initial value for $\delta_d$ in FESD-J reformulation.
    initial_gamma: float          = 1 # double: Initial value for $\gamma$ in FESD-J reformulation.
    initial_beta: float           = 1 # double: Initial value for $\beta$ in FESD-J reformulation.
    initial_p_vt: float           = 1 # double: Initial value for positive tangential velocity slack in FESD-J reformulation.
    initial_n_vt: float           = 1 # double: Initial value for negative tangential velocity slack in FESD-J reformulation.
    initial_alpha_vt: float       = 1 # double: Initial value fo tangential velocity step function in FESD-J reformulation.

    #--------------------- End Initial Values ---------------------#

    #--------------------- Max Values ---------------------#

    ub_lambda_gcs: float  = np.inf # double: Max value for $\lambda$ in the Gradient Comlementarity System.

    ub_Lambda_normal: float  = np.inf # double: Max value for $\Lambda_n$ in FESD-J reformulation.
    ub_P_vn: float           = np.inf # double: Max value for positive normal velocity slack in FESD-J reformulation impulse calculation.
    ub_N_vn: float           = np.inf # double: Max value for negative normal velocity slack in FESD-J reformulation impulse calculation.
    ub_Y_gap: float          = np.inf # double: Max value for gap function in FESD-J reformulation impulse calculation.
    ub_Lambda_tangent: float = np.inf # double: Max value for $\Lambda_t$ in FESD-J reformulation impulse calculation.
    ub_Gamma_d: float        = np.inf # double: Max value for $\Gamma_d$ in FESD-J reformulation impulse calculation.
    ub_Beta_d: float         = np.inf # double: Max value for $\Beta_d$ in FESD-J reformulation impulse calculation.
    ub_Delta_d: float        = np.inf # double: Max value for $\Delta_d$ in FESD-J reformulation impulse calculation.
    ub_Gamma: float          = np.inf # double: Max value for $\Gamma$ in FESD-J reformulation impulse calculation.
    ub_Beta: float           = np.inf # double: Max value for $\Beta$ in FESD-J reformulation impulse calculation.
    ub_P_vt: float           = np.inf # double: Max value for positive tangential velocity slack in FESD-J reformulation impulse calculation.
    ub_N_vt: float           = np.inf # double: Max value for negative tangential velocity slack in FESD-J reformulation impulse calculation.
    ub_Alpha_vt: float       = np.inf # double: Max value fo tangential velocity step function in FESD-J reformulation impulse calculation.

    ub_lambda_normal: float  = np.inf # double: Max value for $\lambda_n$ in FESD-J reformulation.
    ub_p_vn: float           = np.inf # double: Max value for positive normal velocity slack in FESD-J reformulation.
    ub_n_vn: float           = np.inf # double: Max value for negative normal velocity slack in FESD-J reformulation.
    ub_y_gap: float          = np.inf # double: Max value for gap function in FESD-J reformulation.
    ub_lambda_tangent: float = np.inf # double: Max value for $\lambda_t$ in FESD-J reformulation.
    ub_gamma_d: float        = np.inf # double: Max value for $\gamma_d$ in FESD-J reformulation.
    ub_beta_d: float         = np.inf # double: Max value for $\beta_d$ in FESD-J reformulation.
    ub_delta_d: float        = np.inf # double: Max value for $\delta_d$ in FESD-J reformulation.
    ub_gamma: float          = np.inf # double: Max value for $\gamma$ in FESD-J reformulation.
    ub_beta: float           = np.inf # double: Max value for $\beta$ in FESD-J reformulation.
    ub_p_vt: float           = np.inf # double: Max value for positive tangential velocity slack in FESD-J reformulation.
    ub_n_vt: float           = np.inf # double: Max value for negative tangential velocity slack in FESD-J reformulation.
    ub_alpha_vt: float       = np.inf # double: Max value fo tangential velocity step function in FESD-J reformulation.

    #--------------------- End Max Values ---------------------#

    lb_sdf_pts: float = -np.inf;
    ub_sdf_pts: float = np.inf;

    # boolean: If true then the convex multiplier expressions are lifted in the Heaviside step reformulation.
    #
    # Warning:
    #     This is not currently implemented for generic Heaviside step DCS.
    pss_lift_step_functions: bool = 0
    n_depth_step_lifting: int     = 2 # int: Depth to which the Heaviside step convex multipliers are lifted.

    gcs_lift_gap_functions: bool = 1 # boolean: If true the step functions $c(x)$ are lifted in the gradient complementarity system reformulation.

    g_path_at_fe: bool  = 0 # boolean: If true we evaluate nonlinear path constraint at every finte element boundary.
    g_path_at_stg: bool = 0 # boolean: If true evaluate nonlinear path constraint at every stage.
    x_box_at_fe: bool   = 1 # boolean: If true we evaluate box constraint for diff states at every finite element boundary point.

    # boolean: If true we evaluate box constraint for diff states at every stage point.
    #
    # Note:
    #    This is set to zero per default in differential rk mode, as it becomes a linear instead of box constraint.
    x_box_at_stg: bool         = 1
    time_optimal_problem: bool = 0 # boolean: If true for an OCP we automatically reformulate the problem to be time optimal.

    rho_h: float = 1 # double: Weight used in heuristic or relaxed step equilibration modes.

    # StepEquilibrationMode: Which step equilibration mode to use.
    #
    # See Also:
    #     `StepEquilibrationMode` for more details on how each mode works.
    step_equilibration: StepEquilibrationMode = StepEquilibrationMode.HEURISTIC_MEAN
    step_equilibration_sigma: float           = 0.1 # double: Slope at zero for the sigmoid used to rescale the indicator function, nu_ki_rescaled = tanh(nu_ki/step_equilibration_sigma).

    equidistant_control_grid: bool = 1 # boolean: If true each control stage is fixed length.

    time_freezing: bool           = 0 # boolean: Use a time freezing reformulation for the given model.
    time_freezing_inelastic: bool = 0 # boolean: Use the specailized time freezing reformulation for systems with inelastic collisions and friction.

    use_speed_of_time_variables: bool    = 0 # boolean: If true speed of time variables are used for the time freezing reformulation or time optimal problem
    local_speed_of_time_variable: bool   = 0 # boolean: If true then each control stage has a speed of time variable. Otherwise a single speed of time variable is used.
    stagewise_clock_constraint: bool     = 1 # boolean: If true the control grid is fixed with constraints for each control stage.
    impose_terminal_phyisical_time: bool = 1 # boolean: If true the terminal physical time in a time freezing system is constrained to be exactly the desired horizon length.
    s_sot0: float                        = 1 # double: Initial value for speed of time variables.
    s_sot_max: float                     = 25 # double: Maximum for speed of time variables.
    s_sot_min: float                     = 1 # double: Minimum for speed of time variables.
    S_sot_nominal: float                 = 1 # double: Nominal speed of time used for regularizing the speed of time variables.
    rho_sot: float                       = 0 # double: Weight used for the speed of time regularization.

    T_final_max: float  = 1e2 # double: Maximum final time for a time optimal problem.
    T_final_min: float  = 0 # double: Minimum final time for a time optimal problem.


    time_freezing_reduced_model: bool           = 0 # boolean: Analytic reduction of lifter formulation, less algebraic variables (experimental). TODO(@armin) What was this supposed to be?
    time_freezing_hysteresis: bool              = 0
    time_freezing_nonlinear_friction_cone: bool = 1 # boolean: If true we use the nonlinear friction cone, otherwise use polyhedral l_inf approximation.
    time_freezing_quadrature_state: bool        = 0 # boolean: If true make a nonsmooth quadrature state to integrate only if physical time is running.
    time_freezing_lift_forces: bool             = 0 # If true replace $\dot = M(q)^f(q,v,u)$ by $dot = z,  M(q)z - f(q,v,u) = 0$.
    # boolean: Experimental, use $c = \max(c1,c2)$ insetad of $c = c_1c_2$.
    # This is used to reduce the number of switching functions needed to generate the T shaped intersections
    # in inelastic time freezing reformulation.
    time_freezing_nonsmooth_switching_fun: bool = 0
    # boolean: Stabilize auxiliary dynamics in \nabla f_c(q) direction in the style of Baumgartner stabilization.
    stabilizing_q_dynamics: bool                = 0
    # double: Constant used for stabilizing auxiliary dynamics in \nabla f_c(q) direction.
    kappa_stabilizing_q_dynamics: float         = 1e-5

    ############################# NOT IMPLEMENTED
    # FrictionModel: Which Friction model to use for the Complementarity Lagrangian System.
    #
    # Default: :mat:class:`FrictionModel.Conic`
    #
    # See Also:
    #     `FrictionModel` for more details as to the differences between the friction models.
    #friction_model : FrictionModel = FrictionModel.Conic;

    # ConicModelSwitchHandling: Which velocity switch handling mode to use when using the Conic friction model
    #
    # See Also:
    #     `ConicModelSwitchHandling` for more details as to the differences between the switch handling modes.
    #conic_model_switch_handling : ConicModelSwitchHandling = ConicModelSwitchHandling.Abs;

    #kappa_friction_reg : float  = 0; # double: Regularization term in friction equations to avoid large multipliers if no contact happens.

    #lift_velocity_state: bool = 0; # boolean: If true define auxliary algebraic vairable, $dot = z_v$, to avoid symbolic inversion of the inertia matrix.
    #eps_cls: float = 1e-3 # double: enforce $f_c$ at Euler step with h * eps_cls
    #fixed_eps_cls: bool = False # boolean: use fixed step eps_cls instead of a multiple of h.

    # double: The constant radius of relaxation for the friction force which enforces a nonempty interior around zero velocity
    #
    # See Also:
    #     More details can be found in :cite:p:`Nurkanovic2023a`
    #eps_t: float = 1e-7

    # NOTIMPLEMENTED
    # ConstraintRelaxationMode: What (if any) relaxation to apply to the terminal constraints.
    #
    # See Also:
    #    `ConstraintRelaxationMode` for a detailed description of the available relaxation modes.
    relax_terminal_constraint: ConstraintRelaxationMode = ConstraintRelaxationMode.NONE;
    rho_terminal: float  = 1e2; # double: Weight used to penalize terminal constraint violation.

    # NOTIMPLEMENTED
    # ConstraintRelaxationMode: What (if any) relaxation to apply to the path constraints.
    #
    # Warning:
    #    Only implemented for CLS.
    #
    # See Also:
    #    `ConstraintRelaxationMode` for a detailed description of the available relaxation modes.
    #relax_path_constraints: ConstraintRelaxationMode = ConstraintRelaxationMode.NONE;
    #rho_path: float  = 1e2; # double: Weight used to penalize terminal constraint violation.

    # boolean: If True the terminal constraint violation penalty is governed by homotopy parameter.
    #
    # Warning:
    #     This option is currently unimplemented.
    #relax_terminal_constraint_homotopy: bool = 0;

    # ConstraintRelaxationMode: What (if any) relaxation to apply to the terminal/or stage numerical time constraints.
    #
    # See Also:
    #    `ConstraintRelaxationMode` for a detailed description of the available relaxation modes.
    # relax_terminal_numerical_time: ConstraintRelaxationMode = ConstraintRelaxationMode.NONE;
    # rho_terminal_numerical_time: float  = 1e2 # double: Weight used to penalize terminal numerical time violation.

    # boolean: If True the terminal numerical time constraint violation penalty is governed by homotopy parameter
    #
    # Warning:
    #     This option is currently unimplemented
    # relax_terminal_numerical_time_homotopy : bool = 0; # us the homotopy parameter for the penalty.

    # ConstraintRelaxationMode: What (if any) relaxation to apply to the terminal/or stage phyical time constraints.
    #
    # See Also:
    #    `ConstraintRelaxationMode` for a detailed description of the available relaxation modes.
    # relax_terminal_physical_time: ConstraintRelaxationMode = ConstraintRelaxationMode.NONE; # instead of imposing $t(T) = T$, add it as $\ell_1$ penalty term.
    # rho_terminal_physical_time: float  = 1e2 # double: Weight used to penalize terminal physical time violation.

    # boolean: If True the terminal physical time constraint violation penalty is governed by homotopy parameter.
    #
    # Warning:
    #     This option is currently unimplemented.
    # relax_terminal_physical_time_homotopy : bool = 0;

    # ConstraintRelaxationMode: What (if any) relaxation to apply to the impulse equations.
    #
    # See Also:
    #    `ConstraintRelaxationMode` for a detailed description of the available relaxation modes.
    # relax_fesdj_impulse: ConstraintRelaxationMode = ConstraintRelaxationMode.NONE;
    # rho_fesdj_impulse: float = 1e2 # double: Weight used to penalize the impulse equation violation.

    # boolean: If false then the Lagrange term is integrated correctly, otherwise we only evaluate it at the
    # ends of control stages. Setting this to true allows us to do parameter estimation with a nonlinear cost function.
    # This is useful to set to true when implementing a maximum liklihood estimator as in combination with
    # an equidistant grid it allows for fixed time grid for measurements.
    # euler_cost_integration: bool = 0

    # no_initial_impacts: bool = 0 # boolan: If true we disallow impulsive contacts at the beginning of the first control stage.

    # use_previous_solution_as_initial_guess: bool = 0 # boolean: When simulating use the previous step as an initial guess for the current one.
    ################################ NOT IMPLEMENTED #################################

    # int: Level of verbosity that the `nosnoc` reformulator uses.
    #
    # Todo:
    #    @anton, @armin document this better.
    print_level: int = 3

    has_clock_state: bool = 0

    T_val: float = 1
    #p_val

    # Time Freezing constants
    a_n: float = 100;
    k_aux: float = 10;
    time_freezing_Heaviside_lifting: bool = True; # boolean: Exploit the time-freezing PSS structure for tailored lifting in Heaviside reformulation, and drastically reduce the number of  algebraic variables.

    # experimental:
    #---------------------------------------------------------------------#

    use_numerical_clock_state: bool = False # logical: instead of sum of $h$ being used for equidistant control steps use a simple integrated state.

    def time_rescaling(self):
        return (self.time_freezing and self.impose_terminal_phyisical_time) or self.time_optimal_problem;

    def _make_T_h_consistent(self):
        # Handle T, h, h_k etc
        if self.T is not None and self.h_k is None and self.h is None:
            # using T + discretization info
            self.h = self.T/self.N_stages
            self.h_k = [self.h]*self.N_stages
        elif self.T is None and self.h_k is not None and self.h is None:
            # using h_k + discretization info
            # h remains unset
            assert len(self.h_k)==self.N_stages
            self.T = sum(self.h_k)
        elif self.T is None and self.h_k is None and self.h is not None:
            # using h + discretization info
            self.h_k = [self.h]*self.N_stages
            self.T = self.h*self.N_stages
        else:
            # Throw an error
            raise Exception("Please provide exactly one of T, h_k, or h.")


    def __post_init__(self):
        # N_finite_elements always emnds up as a list.
        if isinstance(self.N_finite_elements, int):
            self.N_finite_elements = [self.N_finite_elements]*self.N_stages

        self._make_T_h_consistent()
