"""
Unit tests for Coulomb friction in the Complementarity Lagrangian System.

These cover the model level contract (friction dimensions, D_tangent construction and validation)
and that every combination of friction model, switch handling and cross complementarity mode
produces a structurally consistent MPCC.
"""
import unittest
import warnings

from parameterized import parameterized
import casadi as ca
import numpy as np

import nosnoc as ns


GRAVITY = 9.81


def planar_model(mu=0.3, J_tangent=None, D_tangent=None, n_c=1):
    """Ball(s) on the ground in the plane: n_q = 2*n_c, one gap function per ball."""
    q = ca.SX.sym("q", 2*n_c)
    v = ca.SX.sym("v", 2*n_c)
    f_c = ca.vertcat(*[q[2*i+1] for i in range(n_c)])
    if J_tangent is None and D_tangent is None:
        cols = []
        for i in range(n_c):
            t = np.zeros((2*n_c, 1)); t[2*i, 0] = 1.0
            cols.append(ca.DM(t))
        J_tangent = ca.horzcat(*cols)
    return ns.model.Cls(
        x=ca.vertcat(q, v), x0=np.zeros(4*n_c), M=np.eye(2*n_c),
        f_v=ca.vertcat(*[ca.vertcat(0.0, -GRAVITY) for _ in range(n_c)]),
        f_c=f_c, e=0.0, mu=mu, J_tangent=J_tangent, D_tangent=D_tangent, name="planar")


def spatial_model(mu=0.2, D_tangent=None):
    """Ball on a plane in 3d: the tangent space is two dimensional."""
    q = ca.SX.sym("q", 3)
    v = ca.SX.sym("v", 3)
    return ns.model.Cls(
        x=ca.vertcat(q, v), x0=np.array([0., 0., 1., 2., 1., 0.]), M=np.eye(3),
        f_v=ca.vertcat(0., 0., -GRAVITY), f_c=q[2], e=0.0, mu=mu,
        J_tangent=ca.DM([[1., 0.], [0., 1.], [0., 0.]]), D_tangent=D_tangent, name="spatial")


def build_problem(model, **opt_kwargs):
    defaults = dict(N_stages=1, N_finite_elements=2, n_s=2,
                    rk_scheme=ns.RKScheme.RADAU_IIA, use_fesd=True, T=0.1)
    opts = ns.Options(**(defaults | opt_kwargs))
    dcs = ns.dcs.Cls(model, opts)
    dtp = ns.discrete_time_problem.Cls(dcs, opts)
    dtp.populate_problem()
    return dcs, dtp


class TestFrictionDims(unittest.TestCase):

    def test_planar_dims(self):
        m = planar_model()
        self.assertTrue(m.friction_exists)
        self.assertEqual(m.dims.n_c, 1)
        self.assertEqual(m.dims.n_dim_contact, 2)
        self.assertEqual(m.dims.n_t_conic, 1)
        self.assertEqual(m.dims.n_facets, 2)
        self.assertEqual(m.friction_dims(ns.FrictionModel.POLYHEDRAL), (2, 2))

    def test_spatial_dims(self):
        m = spatial_model()
        self.assertEqual(m.dims.n_dim_contact, 3)
        self.assertEqual(m.dims.n_t_conic, 2)
        self.assertEqual(m.dims.n_facets, 4)
        self.assertEqual(m.friction_dims(ns.FrictionModel.CONIC), (2, 2))
        self.assertEqual(m.friction_dims(ns.FrictionModel.POLYHEDRAL), (4, 4))

    def test_no_friction_leaves_dims_zero(self):
        m = planar_model(mu=0.0)
        self.assertFalse(m.friction_exists)
        self.assertEqual(m.friction_dims(ns.FrictionModel.POLYHEDRAL), (0, 0))
        self.assertEqual(m.friction_dims(ns.FrictionModel.CONIC), (0, 0))


class TestDTangentConstruction(unittest.TestCase):

    def test_autobuild_planar(self):
        m = planar_model()
        np.testing.assert_allclose(np.array(ca.DM(m.D_tangent)), [[1., -1.], [0., 0.]])

    def test_autobuild_spatial(self):
        """Matches the D_tangent of the MATLAB bouncing_ball_3d_sim.m example."""
        m = spatial_model()
        np.testing.assert_allclose(
            np.array(ca.DM(m.D_tangent)),
            [[1., -1., 0., 0.], [0., 0., 1., -1.], [0., 0., 0., 0.]])

    def test_autobuild_blocks_per_contact(self):
        """
        With two contacts the generators of each contact must be contiguous, so that the block
        `ind_i` used by the friction equations really belongs to contact i.
        """
        m = planar_model(n_c=2)
        D = np.array(ca.DM(m.D_tangent))
        self.assertEqual(D.shape, (4, 4))
        np.testing.assert_allclose(D[:, 0:2], [[1., -1.], [0., 0.], [0., 0.], [0., 0.]])
        np.testing.assert_allclose(D[:, 2:4], [[0., 0.], [0., 0.], [1., -1.], [0., 0.]])

    def test_rejects_all_positives_first_ordering(self):
        """
        [J_tangent, -J_tangent] is the ordering used by MATLAB's spinner example. It is fine for a
        single contact but mis-associates generators with contacts as soon as there are two.
        """
        J = ca.DM([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])
        with self.assertRaisesRegex(RuntimeError, "no matching -column"):
            planar_model(n_c=2, J_tangent=J, D_tangent=ca.horzcat(J, -J))

    def test_accepts_all_positives_first_for_single_contact(self):
        J = ca.DM([[1.], [0.]])
        m = planar_model(J_tangent=J, D_tangent=ca.horzcat(J, -J))
        self.assertEqual(m.dims.n_facets, 2)

    def test_rejects_column_without_negation(self):
        with self.assertRaisesRegex(RuntimeError, "no matching -column"):
            planar_model(D_tangent=ca.DM([[1., 0.], [0., 1.]]))

    def test_rejects_odd_facet_count(self):
        with self.assertRaisesRegex(RuntimeError, "even number of columns"):
            planar_model(D_tangent=ca.DM([[1., -1., 0.], [0., 0., 0.]]))

    def test_accepts_evenly_spaced_generators(self):
        """A finer 3d cone whose negation partners are not adjacent columns."""
        n = 8
        angles = 2*np.pi*np.arange(n)/n
        D = ca.DM(np.vstack([np.cos(angles), np.sin(angles), np.zeros(n)]))
        m = spatial_model(D_tangent=D)
        self.assertEqual(m.dims.n_facets, 8)
        self.assertEqual(m.friction_dims(ns.FrictionModel.POLYHEDRAL), (8, 8))

    def test_warns_on_non_unit_generators(self):
        with self.assertWarnsRegex(UserWarning, "not unit vectors"):
            planar_model(D_tangent=ca.DM([[2., -2.], [0., 0.]]))

    def test_warns_on_non_orthonormal_tangent_basis(self):
        with self.assertWarnsRegex(UserWarning, "not orthonormal"):
            spatial_model_J = ca.DM([[1., 1.], [0., 1.], [0., 0.]])
            ns.model.Cls(x=ca.vertcat(ca.SX.sym("q", 3), ca.SX.sym("v", 3)),
                         x0=np.zeros(6), M=np.eye(3), f_v=ca.SX.zeros(3),
                         f_c=ca.SX.sym("q", 3)[2], e=0.0, mu=0.2,
                         J_tangent=spatial_model_J, name="skew")


class TestFrictionValidation(unittest.TestCase):

    def test_friction_without_any_jacobian(self):
        q = ca.SX.sym("q", 2); v = ca.SX.sym("v", 2)
        with self.assertRaisesRegex(RuntimeError, "needs a tangent Jacobian"):
            ns.model.Cls(x=ca.vertcat(q, v), x0=np.zeros(4), M=np.eye(2),
                         f_v=ca.vertcat(0.0, -GRAVITY), f_c=q[1], e=0.0, mu=0.3, name="bad")

    def test_conic_rejected_for_planar_contact(self):
        with self.assertRaisesRegex(RuntimeError, "planar contact"):
            planar_model().friction_dims(ns.FrictionModel.CONIC)

    def test_conic_requires_tangent_basis(self):
        m = planar_model(J_tangent=None, D_tangent=ca.DM([[1., -1.], [0., 0.]]))
        with self.assertRaisesRegex(RuntimeError, "needs the tangent basis"):
            m.friction_dims(ns.FrictionModel.CONIC)

    def test_wrong_row_count(self):
        with self.assertRaisesRegex(RuntimeError, "one row per generalized coordinate"):
            planar_model(J_tangent=ca.DM([[1.], [0.], [0.]]))

    def test_columns_not_divisible_by_contacts(self):
        with self.assertRaisesRegex(RuntimeError, "same number of columns"):
            planar_model(n_c=2, J_tangent=ca.DM(np.eye(4)[:, 0:3]))


class TestProblemStructure(unittest.TestCase):

    @parameterized.expand([
        (fm, sh, ccm)
        for fm in ns.FrictionModel
        for sh in ns.ConicModelSwitchHandling
        for ccm in ns.CrossComplementarityMode
        if fm is ns.FrictionModel.CONIC or sh is ns.ConicModelSwitchHandling.ABS
    ])
    def test_builds_and_pairs_match(self, fm, sh, ccm):
        _, dtp = build_problem(spatial_model(), friction_model=fm,
                               conic_model_switch_handling=sh, cross_comp_mode=ccm)
        self.assertEqual(dtp.G.sym.shape, dtp.H.sym.shape)
        self.assertGreater(dtp.G.sym.shape[0], 0)

    @parameterized.expand([(fm,) for fm in ns.FrictionModel])
    def test_builds_without_fesd(self, fm):
        _, dtp = build_problem(spatial_model(), friction_model=fm, use_fesd=False, n_s=1)
        self.assertEqual(dtp.G.sym.shape, dtp.H.sym.shape)

    def test_builds_relaxed_oc(self):
        _, dtp = build_problem(spatial_model(), friction_model=ns.FrictionModel.POLYHEDRAL,
                               cls_discretization=ns.ClsDiscretization.RELAXED_OC)
        self.assertEqual(dtp.G.sym.shape, dtp.H.sym.shape)

    def test_switch_handling_changes_variable_count(self):
        counts = {}
        for sh in ns.ConicModelSwitchHandling:
            _, dtp = build_problem(spatial_model(), friction_model=ns.FrictionModel.CONIC,
                                   conic_model_switch_handling=sh)
            counts[sh] = dtp.w.sym.shape[0]
        self.assertLess(counts[ns.ConicModelSwitchHandling.PLAIN],
                        counts[ns.ConicModelSwitchHandling.ABS])
        self.assertLess(counts[ns.ConicModelSwitchHandling.ABS],
                        counts[ns.ConicModelSwitchHandling.LP])

    def test_stage_and_impulse_stacks_match_the_dcs(self):
        """
        The discrete time problem rebuilds z_alg / z_impulse from the dcs block lists; if those
        drift apart the wrong variable is fed to the wrong equation without any error.
        """
        dcs, dtp = build_problem(spatial_model(), friction_model=ns.FrictionModel.CONIC)
        self.assertEqual(dtp._build_z_impulse(1, 2).shape[0], dcs.z_impulse.shape[0])
        stage = dtp._get_rk_stage_z(1, 1, 1)
        self.assertEqual(stage.shape[0],
                         dcs.dims.n_x + dcs.dims.n_z + dcs.z_alg.shape[0])

    def test_model_dims_not_mutated_by_dcs(self):
        """
        n_t depends on opts, so it lives on the dcs dims. `Dims.__setattr__` writes through to the
        parent, so a model level n_t would make two discretizations of one model clobber each
        other; this pins that they stay independent.
        """
        model = spatial_model()
        dcs_conic, _ = build_problem(model, friction_model=ns.FrictionModel.CONIC)
        dcs_poly, _ = build_problem(model, friction_model=ns.FrictionModel.POLYHEDRAL)
        # building the second must not have rewritten the first through the Dims parent chain
        self.assertEqual(dcs_conic.dims.n_t, 2)
        self.assertEqual(dcs_poly.dims.n_t, 4)
        self.assertEqual(model.dims.n_t_conic, 2)
        self.assertEqual(model.dims.n_facets, 4)

    @parameterized.expand([(ns.FrictionModel.CONIC, 2), (ns.FrictionModel.POLYHEDRAL, 4)])
    def test_dcs_builds_the_selected_variant(self, friction_model, n_t):
        dcs, _ = build_problem(spatial_model(), friction_model=friction_model)
        self.assertEqual(dcs.dims.n_t, n_t)
        self.assertEqual(dcs.friction_model, friction_model)

    def test_planar_conic_fails_at_dcs_construction(self):
        """The unusable combination is reported when the dcs is built, not as a later shape error."""
        with self.assertRaisesRegex(RuntimeError, "planar contact"):
            build_problem(planar_model(), friction_model=ns.FrictionModel.CONIC)

    def test_frictionless_dcs_has_an_empty_friction_variant(self):
        dcs, _ = build_problem(planar_model(mu=0.0))
        self.assertIsNone(dcs.friction_model)
        self.assertEqual(dcs.dims.n_tangents, 0)
        self.assertEqual(dcs.z_alg_blocks, ["lambda_normal", "y_gap"])

    @parameterized.expand([(f,) for f in ns.ConicModelConeFormulation])
    def test_builds_for_every_cone_formulation(self, formulation):
        dcs, dtp = build_problem(spatial_model(), friction_model=ns.FrictionModel.CONIC,
                                 conic_model_cone_formulation=formulation, eps_t=1e-3)
        self.assertEqual(dcs.cone_formulation, formulation)
        self.assertEqual(dtp.G.sym.shape, dtp.H.sym.shape)

    def test_frictionless_problem_has_no_friction_variables(self):
        _, dtp = build_problem(planar_model(mu=0.0))
        for name in ("lambda_tangent", "gamma", "beta", "gamma_d", "beta_d", "delta_d",
                     "p_vt", "n_vt", "alpha_vt"):
            self.assertNotIn(name, dtp.w.variables)


class TestConeFormulation(unittest.TestCase):
    """The conic friction cone constraints selected by `conic_model_cone_formulation`."""

    MU = 0.3

    @staticmethod
    def cone(formulation, lambda_normal, lambda_tangent, eps, mu=MU):
        return ns.dcs.Cls._cone_constraint(formulation, mu, lambda_normal,
                                           ca.DM(lambda_tangent), eps)

    @parameterized.expand([(f,) for f in ns.ConicModelConeFormulation])
    def test_gradient_matches_the_constraint(self, formulation):
        """The stationarity condition uses the returned gradient, so it must be the real one."""
        lam_n = ca.SX.sym("lambda_n")
        lam_t = ca.SX.sym("lambda_t", 2)
        eps = 1e-2
        cone, grad = ns.dcs.Cls._cone_constraint(formulation, self.MU, lam_n, lam_t, eps)
        f = ca.Function("f", [lam_n, lam_t], [ca.jacobian(cone, lam_t).T, grad])
        rng = np.random.default_rng(0)
        for _ in range(20):
            jac, grad_val = f(rng.uniform(0, 2), rng.uniform(-1, 1, 2))
            np.testing.assert_allclose(np.array(jac), np.array(grad_val), atol=1e-12)

    def test_squared_and_nonsquared_describe_the_same_set(self):
        """Squaring the non-squared constraint gives the squared one with eps_sq = 2*eps/mu."""
        eps = 1e-2
        F = ns.ConicModelConeFormulation
        rng = np.random.default_rng(1)
        for _ in range(200):
            lam_n, lam_t = rng.uniform(0, 1), rng.uniform(-0.4, 0.4, 2)
            g_nonsq, _ = self.cone(F.NONSQUARED, lam_n, lam_t, eps)
            g_sq, _ = self.cone(F.SQUARED, lam_n, lam_t, 2*eps/self.MU)
            if abs(float(g_nonsq)) > 1e-9:
                self.assertEqual(np.sign(float(g_nonsq)), np.sign(float(g_sq)))

    @parameterized.expand([(ns.ConicModelConeFormulation.SQUARED, 0.3**2*1e-2),
                           (ns.ConicModelConeFormulation.NONSQUARED, 0.3)])
    def test_regularized_apex_admits_only_zero_friction(self, formulation, d_cone_d_lambda_n):
        """
        An open contact must still admit exactly lambda_t = 0, and the regularization must give the
        constraint a nonzero gradient there, which is the point of it.
        """
        eps = 1e-2
        g_apex, _ = self.cone(formulation, 0.0, [0.0, 0.0], eps)
        self.assertAlmostEqual(float(g_apex), 0.0, places=14)
        g_off, _ = self.cone(formulation, 0.0, [1e-3, 0.0], eps)
        self.assertLess(float(g_off), 0.0)

        lam_n = ca.SX.sym("lambda_n")
        cone, _ = ns.dcs.Cls._cone_constraint(formulation, self.MU, lam_n, ca.DM([0.0, 0.0]), eps)
        slope = ca.Function("slope", [lam_n], [ca.jacobian(cone, lam_n)])(0.0)
        self.assertAlmostEqual(float(slope), d_cone_d_lambda_n, places=12)

    def test_shifted_cone_forces_friction_at_an_open_contact(self):
        """The shifted cone is a translation: an open contact admits only lambda_t = -eps."""
        eps = 1e-2
        F = ns.ConicModelConeFormulation
        g_zero, _ = self.cone(F.SQUARED_SHIFTED, 0.0, [0.0, 0.0], eps)
        self.assertLess(float(g_zero), 0.0)
        g_shifted, grad = self.cone(F.SQUARED_SHIFTED, 0.0, [-eps, -eps], eps)
        self.assertAlmostEqual(float(g_shifted), 0.0, places=14)
        np.testing.assert_allclose(np.array(grad), 0.0)

    def test_default_is_the_unregularized_squared_cone(self):
        opts = ns.Options(T=1.0)
        self.assertEqual(opts.conic_model_cone_formulation, ns.ConicModelConeFormulation.SQUARED)
        self.assertEqual(opts.eps_t, 0.0)
        g, _ = self.cone(opts.conic_model_cone_formulation, 0.7, [0.1, -0.2], opts.eps_t)
        self.assertAlmostEqual(float(g), (self.MU*0.7)**2 - 0.05, places=14)

    def test_nonsquared_requires_positive_eps(self):
        with self.assertRaisesRegex(RuntimeError, "needs eps_t > 0"):
            build_problem(spatial_model(), friction_model=ns.FrictionModel.CONIC,
                          conic_model_cone_formulation=ns.ConicModelConeFormulation.NONSQUARED)

    def test_negative_eps_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must be nonnegative"):
            build_problem(spatial_model(), friction_model=ns.FrictionModel.CONIC, eps_t=-1e-3)

    def test_eps_is_ignored_by_the_polyhedral_model(self):
        """Validation only applies where eps_t is used."""
        build_problem(spatial_model(), friction_model=ns.FrictionModel.POLYHEDRAL, eps_t=-1.0,
                      conic_model_cone_formulation=ns.ConicModelConeFormulation.NONSQUARED)


if __name__ == "__main__":
    unittest.main()
