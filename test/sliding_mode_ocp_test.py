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
DCS_MODES = [nosnoc.DcsMode.STEWART]
STEP_EQUILIBRATION_MODES = [
    nosnoc.StepEquilibrationMode.HEURISTIC_MEAN, nosnoc.StepEquilibrationMode.HEURISTIC_DELTA,
    nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
]

options = [
    (equidistant_control_grid, step_equilibration, rk_representation, rk_scheme, dcs_mode)
    for equidistant_control_grid in EQUIDISTANT_CONTROLS
    for step_equilibration in STEP_EQUILIBRATION_MODES
    for rk_representation in nosnoc.RKRepresentation
    for rk_scheme in nosnoc.RKScheme
    for dcs_mode in DCS_MODES
]

# test HomotopyUpdateRule.SUPERLINEAR separately without cartesian product
# options += [
#     (True, nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.IrkRepresentation.DIFFERENTIAL,
#      nosnoc.IrkSchemes.RADAU_IIA, nosnoc.DcsMode.STEWART, nosnoc.HomotopyUpdateRule.SUPERLINEAR, nosnoc.MpccMode.SCHOLTES_EQ),
# ]


class TestSlidingModeOcp(unittest.TestCase):

    def test_default(self):
        example(plot=False)

    @parameterized.expand(options)
    def test_combination(self, equidistant_control_grid, step_equilibration, rk_representation,
                         rk_scheme, dcs_mode):
        opts = get_default_options(
            N_stages = 5,
            N_finite_elements = 2,
            equidistant_control_grid = equidistant_control_grid,
            step_equilibration = step_equilibration,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme,
            dcs_mode = dcs_mode
        )
        message = (
            f"Test setting: equidistant_control_grid {equidistant_control_grid}" +
            f"\n{step_equilibration}\n{rk_representation}\n{rk_scheme}\n{dcs_mode}"
        )
        print(message)
        solver = solve_ocp(opts)

        x_traj = solver.get("x")
        u_traj = solver.get("u")
        t_grid = solver.get_time_grid()
        print(f"t_grid = {t_grid}")

        self.assertTrue(np.allclose(x_traj[0,:], X0, atol=1e-4), message)
        self.assertTrue(np.allclose(x_traj[-1,:2], X_TARGET, atol=1e-4), message)
        self.assertTrue(np.allclose(t_grid[-1], TERMINAL_TIME, atol=1e-6), message)
        self.assertTrue(np.allclose(t_grid[0], 0.0, atol=1e-6), message)
        self.assertTrue(np.all(u_traj < UBU), message)
        self.assertTrue(np.all(u_traj > LBU), message)


if __name__ == "__main__":
    unittest.main()
