import unittest
from parameterized import parameterized
import numpy as np
import nosnoc
from examples.cart_pole.main_cart_pole import (
    run_example
)

options = [
    (rk_representation, rk_scheme)
    for rk_representation in nosnoc.RKRepresentation
    for rk_scheme in nosnoc.RKScheme
]

class TestSmoothOcp(unittest.TestCase):

    @parameterized.expand(options)
    def test_combination(self, rk_representation, rk_scheme):
        message = (
            f"rk_representation: \n{rk_representation}" +
            f"rk_scheme \n{rk_scheme}"
        )
        print(message)
        solver = run_example(rk_representation=rk_representation, rk_scheme=rk_scheme)

        x_traj = solver.get("x")
        u_traj = solver.get("u")
        t_grid = solver.get_time_grid()

        print(x_traj[-1,:])
        self.assertTrue(np.allclose(x_traj[-1,1], [np.pi], atol=1e-2), message)
        self.assertTrue(np.allclose(t_grid[0], 0.0, atol=1e-6), message)


if __name__ == "__main__":
    unittest.main()
