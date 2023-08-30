import unittest
from parameterized import parameterized
import numpy as np
import nosnoc
from examples.simplest.simplest_example import (
    get_default_options,
    get_simplest_model_sliding,
    solve_simplest_example
    )
IRK_SCHEMES = [
    nosnoc.IrkSchemes.RADAU_IIA,
    nosnoc.IrkSchemes.GAUSS_LEGENDRE
]
IRK_REPRESENTATIONS = [
    nosnoc.IrkRepresentation.DIFFERENTIAL,
    nosnoc.IrkRepresentation.DIFFERENTIAL_LIFT_X,
    nosnoc.IrkRepresentation.INTEGRAL
]

PSS_MODES = [
    nosnoc.PssMode.STEP,
    nosnoc.PssMode.STEWART
]

CROSS_COMP_MODES = [
    nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER,
    nosnoc.CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
]

options = [
    (irk_scheme, irk_representation, pss_mode, cross_comp_mode)
    for irk_scheme in IRK_SCHEMES
    for irk_representation in IRK_REPRESENTATIONS
    for pss_mode in PSS_MODES
    for cross_comp_mode in CROSS_COMP_MODES
]


class TestSlidingMode(unittest.TestCase):

    @parameterized.expand(options)
    def test_combination(self, irk_scheme, irk_representation, pss_mode, cross_comp_mode):
        opts = get_default_options()
        opts.comp_tol = 1e-7
        opts.irk_representation = irk_representation
        opts.irk_scheme = irk_scheme
        opts.pss_mode = pss_mode
        opts.cross_comp_mode = cross_comp_mode

        message = (
            f"Test setting:\n{irk_representation}\n{irk_scheme}\n{pss_mode}\n{cross_comp_mode}"
        )
        print(message)
        x0 = np.array([-np.sqrt(2)])
        results = solve_simplest_example(opts, get_simplest_model_sliding(x0), x0=x0, Nsim=7, Tsim=2.0)

        x_traj = results["X_sim"]
        t_grid = results["t_grid"]
        print(x_traj)
        print(t_grid)
        xt = np.vstack((x_traj.T, t_grid))
        diff = xt - np.array([[0], [np.sqrt(2)]])

        self.assertTrue(np.min(np.linalg.norm(diff, axis=0)) < 1e-5, message)


if __name__ == "__main__":
    unittest.main()
