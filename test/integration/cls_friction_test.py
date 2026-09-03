"""
Integration tests for Coulomb friction in the CLS, against closed form solutions.

Both test problems have an exact solution: a ball is dropped with a tangential velocity, takes one
inelastic impact whose friction impulse is capped by mu*Lambda_n, and then slides under a constant
friction force. That exercises the friction force, the friction impulse and the cone bound at once.
"""
import unittest
import warnings

from parameterized import parameterized
import numpy as np

import nosnoc as ns

from examples.cls_minimal_example.bouncing_ball_2d import (
    analytic_solution as analytic_2d,
    get_default_options as opts_2d,
    solve_bouncing_ball_2d,
    MU as MU_2D,
    X0 as X0_2D,
)
from examples.cls_minimal_example.bouncing_ball_3d import (
    analytic_solution as analytic_3d,
    get_default_options as opts_3d,
    solve_bouncing_ball_3d,
    MU as MU_3D,
)
from examples.cls_minimal_example.bouncing_ball_1d import (
    solve_bouncing_ball as solve_1d,
    analytic_solution as analytic_1d,
)


class TestPlanarFriction(unittest.TestCase):
    """In the plane the polyhedral cone is exact, so the analytic solution must be matched."""

    def test_matches_analytic(self):
        t_grid, x_res, _ = solve_bouncing_ball_2d(mu=MU_2D)
        _, qx_a, _, vx_a, _, _, _ = analytic_2d(MU_2D)
        self.assertAlmostEqual(x_res[-1, 0], qx_a[-1], places=5)
        self.assertAlmostEqual(x_res[-1, 2], vx_a[-1], places=5)
        self.assertAlmostEqual(x_res[-1, 1], 0.0, places=6)

    def test_friction_impulse_is_capped_by_the_cone(self):
        _, _, integrator = solve_bouncing_ball_2d(mu=MU_2D)
        Lambda_n = integrator.get("Lambda_normal").flatten()
        Lambda_t = integrator.get("Lambda_tangent").reshape(-1, 2)
        # Polyhedral: the budget sum(Lambda_t) may not exceed mu*Lambda_n.
        self.assertTrue(np.all(Lambda_t >= -1e-7))
        np.testing.assert_array_less(Lambda_t.sum(axis=1), MU_2D*Lambda_n + 1e-6)

    def test_contact_force_respects_the_cone(self):
        _, _, integrator = solve_bouncing_ball_2d(mu=MU_2D)
        lam_n = integrator.get_full("lambda_normal").flatten()
        lam_t = integrator.get_full("lambda_tangent").reshape(-1, 2)
        np.testing.assert_array_less(lam_t.sum(axis=1), MU_2D*lam_n + 1e-6)

    def test_more_friction_stops_the_ball_sooner(self):
        finals = []
        for mu in (0.1, 0.3, 0.6):
            _, x_res, _ = solve_bouncing_ball_2d(mu=mu)
            finals.append(x_res[-1, 0])
        self.assertTrue(finals[0] > finals[1] > finals[2])

    def test_zero_friction_slides_forever(self):
        """mu = 0 disables friction entirely, so the tangential velocity is preserved."""
        _, x_res, _ = solve_bouncing_ball_2d(mu=0.0)
        self.assertAlmostEqual(x_res[-1, 2], X0_2D[2], places=6)

    def test_conic_is_rejected_in_the_plane(self):
        with self.assertRaisesRegex(RuntimeError, "planar contact"):
            solve_bouncing_ball_2d(
                mu=MU_2D, opts=opts_2d(friction_model=ns.FrictionModel.CONIC))


class TestSpatialFriction(unittest.TestCase):

    def test_conic_matches_analytic(self):
        q_a, v_a, _, _ = analytic_3d(MU_3D)
        _, x_res, _ = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        np.testing.assert_allclose(x_res[-1, 0:2], q_a, atol=1e-3)
        np.testing.assert_allclose(x_res[-1, 3:5], v_a, atol=1e-3)

    def test_conic_preserves_the_sliding_direction(self):
        """Isotropic friction opposes the sliding direction, so 2:1 must stay 2:1."""
        _, x_res, _ = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        self.assertAlmostEqual(x_res[-1, 3]/x_res[-1, 4], 2.0, places=3)

    def test_conic_impulse_respects_the_cone(self):
        _, _, integrator = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        Lambda_n = integrator.get("Lambda_normal").flatten()
        Lambda_t = integrator.get("Lambda_tangent").reshape(-1, 2)
        np.testing.assert_array_less(np.linalg.norm(Lambda_t, axis=1), MU_3D*Lambda_n + 1e-5)

    def test_conic_produces_a_single_impact_impulse(self):
        """
        A single drop must give a single impulse. A formulation whose multipliers blow up while the
        contact is open smears it over several finite elements instead.
        """
        _, _, integrator = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        Lambda_n = integrator.get("Lambda_normal").flatten()
        self.assertEqual(int(np.sum(np.abs(Lambda_n) > 1e-3)), 1)

    def test_conic_multipliers_are_well_scaled_in_contact(self):
        """
        While the contact carries force, gamma is the multiplier of the cone constraint and the
        scaled stationarity condition fixes it at |v_t|/2, so it must stay O(1). Without the cone
        radius scaling it instead runs away to ~1e8 and drags the impulses with it.

        While the contact is *open* gamma is genuinely indeterminate: it only ever appears
        multiplied by lambda_tangent, which is zero there, so any value solves the equation and the
        solver parks it arbitrarily. That is asserted separately below rather than bounded here.
        """
        _, _, integrator = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        gamma = integrator.get_full("gamma").flatten()
        lam_n = integrator.get_full("lambda_normal").flatten()
        in_contact = lam_n > 1e-3
        self.assertTrue(np.any(in_contact))
        # |v_t| never exceeds its initial sqrt(5), so gamma = |v_t|/2 stays below ~1.2.
        self.assertLess(gamma[in_contact].max(), 5.0)

    def test_conic_friction_force_vanishes_out_of_contact(self):
        """No tangential force may act while the ball is in free flight."""
        _, _, integrator = solve_bouncing_ball_3d(friction_model=ns.FrictionModel.CONIC)
        lam_n = integrator.get_full("lambda_normal").flatten()
        lam_t = integrator.get_full("lambda_tangent").reshape(-1, 2)
        open_contact = lam_n < 1e-6
        self.assertTrue(np.any(open_contact))
        self.assertLess(np.abs(lam_t[open_contact]).max(), 1e-8)

    def test_polyhedral_converges_to_the_conic_answer(self):
        """
        The n-gon inscribed in the friction disc underestimates friction by 1-cos(pi/n), so a finer
        cone must track the exact conic answer more closely.
        """
        _, v_a, _, _ = analytic_3d(MU_3D)
        speed_a = np.linalg.norm(v_a)
        errors = []
        for n_facets in (None, 8, 16):
            _, x_res, _ = solve_bouncing_ball_3d(
                friction_model=ns.FrictionModel.POLYHEDRAL, n_facets=n_facets)
            errors.append(abs(np.linalg.norm(x_res[-1, 3:5]) - speed_a))
        self.assertTrue(errors[0] > errors[1] > errors[2],
                        f"expected monotone refinement, got {errors}")

    @parameterized.expand([(sh,) for sh in ns.ConicModelSwitchHandling])
    def test_all_switch_handlings_solve(self, sh):
        _, x_res, _ = solve_bouncing_ball_3d(
            opts=opts_3d(friction_model=ns.FrictionModel.CONIC,
                         conic_model_switch_handling=sh))
        self.assertAlmostEqual(x_res[-1, 2], 0.0, places=5)


class TestFrictionlessRegression(unittest.TestCase):
    """
    The friction work refactored the variable stacking and the switch indicator, both of which the
    frictionless CLS also goes through. These pin that its results did not move.
    """

    @parameterized.expand([(0.0,), (0.9,)])
    def test_bouncing_ball_1d_unchanged(self, e):
        _, x_res, _ = solve_1d(e=e)
        _, q_a, v_a, _ = analytic_1d(e)
        self.assertAlmostEqual(x_res[-1, 0], q_a[-1], places=4)
        self.assertAlmostEqual(x_res[-1, 1], v_a[-1], places=4)


if __name__ == "__main__":
    unittest.main()
