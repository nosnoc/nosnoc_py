import unittest
from parameterized import parameterized
from examples.motor_with_friction_ocp import (
    solve_ocp,
    example,
    get_default_options,
    X0,
    X_TARGET,
)
import nosnoc
import numpy as np

options = [(step_equilibration, pss_mode)
           for pss_mode in nosnoc.PssMode
           for step_equilibration in nosnoc.StepEquilibrationMode
           if step_equilibration != nosnoc.StepEquilibrationMode.DIRECT]


class TestOcpMotor(unittest.TestCase):

    def test_default(self):
        example(plot=False)

    @parameterized.expand(options)
    def test_problem(self, step_equilibration, pss_mode):
        opts = get_default_options()
        opts.step_equilibration = step_equilibration
        opts.pss_mode = pss_mode
        # opts.print_level = 0

        # print(f"test setting: {opts}")
        results = solve_ocp(opts)

        x_traj = results["x_traj"]

        message = f"For step_equilibration {step_equilibration} and pss_mode {pss_mode}"
        self.assertTrue(np.allclose(x_traj[0], X0, atol=1e-4), message)
        self.assertTrue(np.allclose(x_traj[-1], X_TARGET, atol=1e-4), message)


if __name__ == "__main__":
    unittest.main()
