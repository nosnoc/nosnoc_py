import unittest
from parameterized import parameterized
import numpy as np
import nosnoc
from examples.sliding_mode_ocp.sliding_mode_ocp import (
    solve_ocp,
    example,
    get_default_options,
    X0,
    X_TARGET,
    TERMINAL_TIME,
    UBU,
    LBU,
)

EQUIDISTANT_CONTROLS = [True, False]
PSS_MODES = [nosnoc.PssMode.STEWART]
options = [
    (equidistant_control_grid, step_equilibration, irk_representation, irk_scheme, pss_mode,
     nosnoc.HomotopyUpdateRule.LINEAR) for equidistant_control_grid in EQUIDISTANT_CONTROLS
    for step_equilibration in [
        nosnoc.StepEquilibrationMode.HEURISTIC_MEAN, nosnoc.StepEquilibrationMode.HEURISTIC_DELTA,
        nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
    ] for irk_representation in nosnoc.IrkRepresentation for irk_scheme in nosnoc.IrkSchemes
    for pss_mode in PSS_MODES
    # Ignore the following cases that currently fail:
    if (equidistant_control_grid, step_equilibration, irk_representation, irk_scheme, pss_mode)
    not in [
        # (True, nosnoc.StepEquilibrationMode.DIRECT, nosnoc.IrkRepresentation.DIFFERENTIAL_LIFT_X, nosnoc.IrkSchemes.RADAU_IIA, nosnoc.PssMode.STEWART, nosnoc.HomotopyUpdateRule.LINEAR),
    ]
]

# test HomotopyUpdateRule.SUPERLINEAR separately without cartesian product
options += [
    (True, nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.IrkRepresentation.DIFFERENTIAL,
     nosnoc.IrkSchemes.RADAU_IIA, nosnoc.PssMode.STEWART, nosnoc.HomotopyUpdateRule.SUPERLINEAR),
]


class TestOcp(unittest.TestCase):

    def test_default(self):
        example(plot=False)

    @parameterized.expand(options)
    def test_combination(self, equidistant_control_grid, step_equilibration, irk_representation,
                         irk_scheme, pss_mode, homotopy_update_rule):
        opts = get_default_options()
        opts.comp_tol = 1e-5
        opts.N_stages = 5
        opts.N_finite_elements = 2
        opts.equidistant_control_grid = equidistant_control_grid
        opts.step_equilibration = step_equilibration
        opts.irk_representation = irk_representation
        opts.irk_scheme = irk_scheme
        opts.pss_mode = pss_mode
        opts.homotopy_update_rule = homotopy_update_rule

        print(
            f"test setting: equidistant_control_grid {equidistant_control_grid}" +
            f"\n{step_equilibration}\n{irk_representation}\n{irk_scheme}\n{pss_mode}\n{homotopy_update_rule}"
        )
        results = solve_ocp(opts)

        x_traj = results["x_traj"]
        u_traj = results["u_traj"]
        t_grid = results["t_grid"]

        message = (
            f"For parameters: control grid {equidistant_control_grid} step_equilibration {step_equilibration}, irk_representation {irk_representation}, "
            f"irk_scheme {irk_scheme}, pss_mode {pss_mode}, homotopy_update_rule {homotopy_update_rule}"
        )

        self.assertTrue(np.allclose(x_traj[0], X0, atol=1e-4), message)
        self.assertTrue(np.allclose(x_traj[-1][:2], X_TARGET, atol=1e-4), message)
        self.assertTrue(np.allclose(t_grid[-1], TERMINAL_TIME, atol=1e-6), message)
        self.assertTrue(np.allclose(t_grid[0], 0.0, atol=1e-6), message)
        self.assertTrue(np.alltrue(u_traj < UBU), message)
        self.assertTrue(np.alltrue(u_traj > LBU), message)


if __name__ == "__main__":
    unittest.main()
