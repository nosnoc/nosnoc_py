import unittest
from examples.cart_pole_with_friction.parametric_cart_pole_with_friction import solve_paramteric_example
from examples.cart_pole_with_friction.cart_pole_with_friction import solve_example
import numpy as np


class TestParametericOcp(unittest.TestCase):

    def test_one_parametric_ocp(self):
        ref_results = solve_example()
        results_parametric = solve_paramteric_example(with_global_var=False)

        self.assertTrue(np.allclose(ref_results["w_sol"], results_parametric["w_sol"], atol=1e-7))
        self.assertTrue(np.alltrue(ref_results["nlp_iter"] == results_parametric["nlp_iter"]))
        self.assertEqual(results_parametric["v_global"].shape, (1, 0))
        self.assertEqual(ref_results["v_global"].shape, (1, 0))

        results_with_global_var = solve_paramteric_example(with_global_var=True)
        self.assertTrue(np.allclose(np.ones((1,)), results_with_global_var["v_global"], atol=1e-7))

        self.assertTrue(
            np.allclose(np.array(ref_results["cost_val"]),
                        np.array(results_with_global_var["cost_val"]),
                        atol=1e-6))


if __name__ == "__main__":
    unittest.main()
