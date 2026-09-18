"""
Integration test for the ellipse squeeze OCP.

There is no closed form solution here, so the test checks the properties that the formulation must
have no matter what the optimizer finds: the homotopy converges, the ellipse actually rises, the
balls never penetrate it, and the friction cone is respected. Both control parametrizations are
covered, since the impedance mode changes the dynamics and the units of the control bounds.

The horizon is shortened relative to the example so that the test stays quick.
"""
import sys
import unittest
from pathlib import Path

from parameterized import parameterized
import numpy as np

# The example is written to be runnable as a plain script, so it imports its own geometry module by
# bare name. That only resolves when its directory is on sys.path, which running it as a script does
# and pytest does not, hence this insertion.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"examples"/"impedance_grasp"))

from ellipse_squeeze import (contact_forces, gap_values, get_default_options,  # noqa: E402
                             get_default_solver_options, solve)
from geometry import GraspConfig  # noqa: E402

N_STAGES = 10
T_SHORT = 0.4


def solve_short(impedance):
    cfg = GraspConfig(impedance=impedance, T=T_SHORT)
    opts = get_default_options(cfg, N_stages=N_STAGES)
    solver_opts = get_default_solver_options(print_level=0)
    return solve(cfg, opts=opts, solver_opts=solver_opts)


class TestEllipseSqueeze(unittest.TestCase):

    @parameterized.expand([("direct", False), ("impedance", True)])
    def test_lifts_without_violating_contact(self, _, impedance):
        solver, cfg, stats = solve_short(impedance)

        self.assertTrue(stats["converged"])

        x = solver.get("x")
        self.assertGreater(x[-1, 1], x[0, 1], "the ellipse should end up higher than it started")
        self.assertLess(abs(x[-1, 2]), 0.1, "the ellipse should stay roughly level")

        # No penetration. The gap is a level set rather than a distance, but its sign is what the
        # complementarity enforces.
        self.assertGreater(gap_values(solver, cfg).min(), -1e-7)

        # Coulomb cone, on the forces that the discretization actually applied.
        lambda_n, lambda_t = contact_forces(solver)
        self.assertGreater(np.min(cfg.mu*lambda_n - np.abs(lambda_t)), -1e-6)
        self.assertGreater(lambda_n.max(), 0.0, "the fingers have to squeeze to lift anything")

    def test_impedance_target_is_pushed_into_the_ball(self):
        """
        The grip force of an impedance controlled finger is K times the tracking error, so lifting
        the ellipse requires the desired position to lie *inside* the ball it commands.
        """
        solver, cfg, _ = solve_short(impedance=True)
        u = np.atleast_2d(solver.get("u"))
        x = solver.get("x")[::2]
        n_fe = solver.opts.N_finite_elements[0]

        # Compare each control with the ball position at the start of its stage.
        ball1_x = x[:-1:n_fe, 3][:u.shape[0]]
        ball2_x = x[:-1:n_fe, 5][:u.shape[0]]
        self.assertLess(np.min(u[:, 0] - ball1_x), -1e-3, "right target must press to the left")
        self.assertGreater(np.max(u[:, 2] - ball2_x), 1e-3, "left target must press to the right")


if __name__ == "__main__":
    unittest.main()
