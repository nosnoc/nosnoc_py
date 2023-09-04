from typing import Union, Optional
from dataclasses import dataclass, field

import numpy as np

from .rk_utils import generate_butcher_tableu, generate_butcher_tableu_integral
from .utils import validate
from .nosnoc_types import MpccMode, IrkSchemes, StepEquilibrationMode, CrossComplementarityMode, IrkRepresentation, PssMode, IrkRepresentation, HomotopyUpdateRule, InitializationStrategy, ConstraintHandling


def _assign(dictionary, keys, value):
    """Assign a value."""
    for key in keys:
        dictionary[key] = value


@dataclass
class NosnocOpts:

    # discretization
    terminal_time: Union[float, int] = 1.0  # TODO: make param?

    use_fesd: bool = True  #: Selects use of fesd or normal RK formulation.
    print_level: int = 0  #: higher -> more info
    max_iter_homotopy: int = 0

    initialization_strategy: InitializationStrategy = InitializationStrategy.ALL_XCURRENT_W0_START

    irk_representation: IrkRepresentation = IrkRepresentation.INTEGRAL

    # IRK and FESD opts
    n_s: int = 2  #: Number of IRK stages
    irk_scheme: IrkSchemes = IrkSchemes.RADAU_IIA
    cross_comp_mode: CrossComplementarityMode = CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
    mpcc_mode: MpccMode = MpccMode.SCHOLTES_INEQ
    constraint_handling: ConstraintHandling = ConstraintHandling.EXACT

    pss_mode: PssMode = PssMode.STEWART  # possible options: Stewart and Step

    use_upper_bound_h: bool = True
    gamma_h: float = 1.0

    smoothing_parameter: float = 1e1  #: used for smoothed Step representation
    # used in InitializationStrategy.RK4_smoothed
    fix_active_set_fe0: bool = False

    N_finite_elements: int = 2  #
    Nfe_list: list = field(default_factory=list)  #: list of length N_stages, Nfe per stage

    # MPCC and Homotopy opts
    comp_tol: float = 1e-8  #: complementarity tolerance
    sigma_0: float = 1.0
    sigma_N: float = 1e-8
    homotopy_update_slope: float = 0.1
    homotopy_update_exponent: float = 1.5
    homotopy_update_rule: HomotopyUpdateRule = HomotopyUpdateRule.LINEAR

    # step equilibration
    step_equilibration: StepEquilibrationMode = StepEquilibrationMode.HEURISTIC_DELTA
    step_equilibration_sigma: float = 0.1
    rho_h: float = 1.0

    # for MpccMode.FISCHER_BURMEISTER_IP_AUG
    fb_ip_aug1_weight: float = 1e0
    fb_ip_aug2_weight: float = 1e-1

    # polishing step
    do_polishing_step: bool = False

    # OCP only
    N_stages: int = 1
    equidistant_control_grid: bool = True  # NOTE: tested in test_ocp

    s_elastic_0: float = 1.0
    s_elastic_max: float = 1e1
    s_elastic_min: float = 0.0

    # IPOPT opts
    opts_casadi_nlp = dict()
    opts_casadi_nlp['print_time'] = 0
    opts_casadi_nlp['verbose'] = False
    opts_casadi_nlp['ipopt'] = dict()
    opts_casadi_nlp['ipopt']['sb'] = 'yes'
    opts_casadi_nlp['ipopt']['max_iter'] = 500
    opts_casadi_nlp['ipopt']['print_level'] = 0
    opts_casadi_nlp['ipopt']['bound_relax_factor'] = 0
    tol_ipopt = property(
        fget=lambda s: s.opts_casadi_nlp['ipopt']['tol'],
        fset=lambda s, v: _assign(
            s.opts_casadi_nlp['ipopt'],
            ['tol', 'dual_inf_tol', 'compl_inf_tol', 'mu_target'], v
        ),
        doc="Ipopt tolerance."
    )
    opts_casadi_nlp['ipopt']['tol'] = None
    opts_casadi_nlp['ipopt']['dual_inf_tol'] = None
    opts_casadi_nlp['ipopt']['compl_inf_tol'] = None
    opts_casadi_nlp['ipopt']['mu_strategy'] = 'adaptive'
    opts_casadi_nlp['ipopt']['mu_oracle'] = 'quality-function'
    opts_casadi_nlp["record_time"] = True
    # opts_casadi_nlp['ipopt']['linear_solver'] = 'ma27'
    # opts_casadi_nlp['ipopt']['linear_solver'] = 'ma57'

    time_freezing: bool = False
    time_freezing_tolerance: float = 1e-3

    rootfinder_for_initial_z: bool = False

    # Usabillity:
    nlp_max_iter = property(
        fget=lambda s: s.opts_casadi_nlp["ipopt"]["max_iter"],
        fset=lambda s, v: _assign(s.opts_casadi_nlp["ipopt"], ["max_iter"], v),
        doc="Maximum amount of iterations for the subsolver."
    )

    def __repr__(self) -> str:
        out = ''
        for k, v in self.__dict__.items():
            out += f"{k} : {v}\n"
        return out

    def preprocess(self):
        # IPOPT tol should be smaller than outer tol, but not too much
        # Note IPOPT option list: https://coin-or.github.io/Ipopt/OPTIONS.html
        if self.tol_ipopt is None:
            self.tol_ipopt = self.comp_tol * 1e-2

        validate(self)
        # self.opts_casadi_nlp['ipopt']['print_level'] = self.print_level

        if self.time_freezing:
            if self.n_s < 3:
                Warning("Problem might be illposed if n_s < 3 and time freezing")
            # TODO: Extend checks

        if self.max_iter_homotopy == 0:
            self.max_iter_homotopy = int(np.round(np.abs(np.log(self.comp_tol / self.sigma_0) / np.log(self.homotopy_update_slope)))) + 1

        if len(self.Nfe_list) == 0:
            self.Nfe_list = self.N_stages * [self.N_finite_elements]

        # Butcher Tableau
        if self.irk_representation == IrkRepresentation.INTEGRAL:
            B_irk, C_irk, D_irk, irk_time_points = generate_butcher_tableu_integral(
                self.n_s, self.irk_scheme)
            self.B_irk = B_irk
            self.C_irk = C_irk
            self.D_irk = D_irk
        elif (self.irk_representation
              in [IrkRepresentation.DIFFERENTIAL, IrkRepresentation.DIFFERENTIAL_LIFT_X]):
            A_irk, b_irk, irk_time_points, _ = generate_butcher_tableu(self.n_s, self.irk_scheme)
            self.A_irk = A_irk
            self.b_irk = b_irk

        self.irk_time_points = irk_time_points
        if np.abs(irk_time_points[-1] - 1.0) < 1e-9:
            self.right_boundary_point_explicit = True
        else:
            self.right_boundary_point_explicit = False

        # checks:
        if (self.cross_comp_mode == CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
                and self.mpcc_mode
                in [MpccMode.FISCHER_BURMEISTER, MpccMode.FISCHER_BURMEISTER_IP_AUG]):
            Warning(
                "UNSUPPORTED option combination comp_mode: SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA and mpcc_mode: MpccMode.FISCHER_BURMEISTER"
            )
        if self.mpcc_mode == MpccMode.FISCHER_BURMEISTER and self.constraint_handling != ConstraintHandling.LEAST_SQUARES:
            Warning(
                "UNSUPPORTED option combination comp_mode: mpcc_mode == MpccMode.FISCHER_BURMEISTER and constraint_handling != ConstraintHandling.LEAST_SQUARES"
            )
        if (self.step_equilibration
                in [StepEquilibrationMode.DIRECT, StepEquilibrationMode.DIRECT_COMPLEMENTARITY] and
                self.constraint_handling != ConstraintHandling.LEAST_SQUARES):
            Warning(
                "UNSUPPORTED option combination: StepEquilibrationMode.DIRECT* and constraint_handling != ConstraintHandling.LEAST_SQUARES"
            )
        return

    ## Options in matlab..
    # MPCC related, not implemented yet.
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
