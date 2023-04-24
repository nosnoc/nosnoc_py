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
MPCC_MODES = [
    nosnoc.MpccMode.SCHOLTES_INEQ,
    nosnoc.MpccMode.SCHOLTES_EQ,
    nosnoc.MpccMode.ELASTIC_INEQ,
    nosnoc.MpccMode.ELASTIC_EQ,
    nosnoc.MpccMode.ELASTIC_TWO_SIDED
]
STEP_EQUILIBRATION_MODES = [
    nosnoc.StepEquilibrationMode.HEURISTIC_MEAN, nosnoc.StepEquilibrationMode.HEURISTIC_DELTA,
    nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
]

options = [
    (equidistant_control_grid, step_equilibration, irk_representation, irk_scheme, pss_mode,
     nosnoc.HomotopyUpdateRule.LINEAR, nosnoc.MpccMode.SCHOLTES_INEQ)
    for equidistant_control_grid in EQUIDISTANT_CONTROLS
    for step_equilibration in STEP_EQUILIBRATION_MODES
    for irk_representation in nosnoc.IrkRepresentation
    for irk_scheme in nosnoc.IrkSchemes
    for pss_mode in PSS_MODES
]

# test MpccMode separately without cartesian product
options = [
    (True, nosnoc.StepEquilibrationMode.HEURISTIC_MEAN, nosnoc.IrkRepresentation.DIFFERENTIAL, nosnoc.IrkSchemes.RADAU_IIA, nosnoc.PssMode.STEWART,
     nosnoc.HomotopyUpdateRule.LINEAR, mpcc_mode)
    for mpcc_mode in MPCC_MODES
]

# test HomotopyUpdateRule.SUPERLINEAR separately without cartesian product
# options += [
#     (True, nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.IrkRepresentation.DIFFERENTIAL,
#      nosnoc.IrkSchemes.RADAU_IIA, nosnoc.PssMode.STEWART, nosnoc.HomotopyUpdateRule.SUPERLINEAR, nosnoc.MpccMode.SCHOLTES_EQ),
# ]


class TestOcp(unittest.TestCase):

    def test_default(self):
        example(plot=False)

    @parameterized.expand(options)
    def test_combination(self, equidistant_control_grid, step_equilibration, irk_representation,
                         irk_scheme, pss_mode, homotopy_update_rule, mpcc_mode):
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
        opts.mpcc_mode = mpcc_mode

        message = (
            f"Test setting: equidistant_control_grid {equidistant_control_grid}" +
            f"\n{step_equilibration}\n{irk_representation}\n{irk_scheme}\n{pss_mode}\n{homotopy_update_rule}"
            f"\n{mpcc_mode}"
        )
        print(message)
        results = solve_ocp(opts)

        x_traj = results["x_traj"]
        u_traj = results["u_traj"]
        t_grid = results["t_grid"]

        self.assertTrue(np.allclose(x_traj[0], X0, atol=1e-4), message)
        self.assertTrue(np.allclose(x_traj[-1][:2], X_TARGET, atol=1e-4), message)
        self.assertTrue(np.allclose(t_grid[-1], TERMINAL_TIME, atol=1e-6), message)
        self.assertTrue(np.allclose(t_grid[0], 0.0, atol=1e-6), message)
        self.assertTrue(np.alltrue(u_traj < UBU), message)
        self.assertTrue(np.alltrue(u_traj > LBU), message)


if __name__ == "__main__":
    unittest.main()
