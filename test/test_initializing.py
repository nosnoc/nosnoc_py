import numpy as np
import unittest
import nosnoc
from examples.sliding_mode_ocp import get_sliding_mode_ocp_description, get_default_options, TERMINAL_TIME


class TestInitialization(unittest.TestCase):
    """Test several initialization methods."""

    def test_initialization(self):
        """Test initialization of x and u."""
        global X0
        opts = get_default_options()
        opts.terminal_time = TERMINAL_TIME
        opts2 = get_default_options()
        opts2.terminal_time = TERMINAL_TIME
        [model, ocp] = get_sliding_mode_ocp_description()
        [model2, ocp2] = get_sliding_mode_ocp_description()
        u1_est = np.zeros((opts.N_stages, 1))
        u1_est[-1] = -2.5
        u1_est[-2] = 0.1
        u2_est = np.zeros((opts.N_stages, 1))
        u2_est[-1] = 0.5
        u2_est[-2] = -2.0
        u = np.concatenate((u1_est, u2_est), axis=1)
        X0 = X0.reshape((1, -1))
        x_est = np.concatenate((
            X0 * 2 / 3, X0 * 1 / 3, np.zeros((opts.N_stages - 2, 4))
        ))

        opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
        solver = nosnoc.NosnocSolver(opts, model, ocp)
        solver2 = nosnoc.NosnocSolver(opts2, model2, ocp2)
        solver.set("x", x_est)
        for i in range(opts.N_stages):
            self.assertTrue(np.array_equal(
                solver.problem.w0[solver.problem.ind_x[i][0][0]], x_est[i, :]
            ))

        solver.set("u", u)
        self.assertTrue(np.array_equal(
            solver.problem.w0[solver.problem.ind_u], u
        ))

        res_initialized = solver.solve()
        res_uninitialized = solver2.solve()
        # If initialized, the iteration should be faster
        self.assertLessEqual(
            res_initialized["nlp_iter"][0], res_uninitialized["nlp_iter"][0]
        )


if __name__ == "__main__":
    unittest.main()
