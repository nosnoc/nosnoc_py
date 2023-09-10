import numpy as np
import unittest
import nosnoc
from examples.sliding_mode_ocp.sliding_mode_ocp import get_sliding_mode_ocp_description, get_default_options, X0, TERMINAL_TIME


class TestInitialization(unittest.TestCase):
    """Test several initialization methods."""

    def test_initialization(self):
        """Test initialization of x and u."""
        global X0
        opts = get_default_options()
        opts.terminal_time = TERMINAL_TIME
        opts.sigma_0 = 1e-1
        opts.comp_tol = 1e-6

        opts2 = get_default_options()
        opts2.terminal_time = TERMINAL_TIME
        opts2.sigma_0 = 1e-1
        opts2.comp_tol = 1e-6

        [model, ocp] = get_sliding_mode_ocp_description()
        [model2, ocp2] = get_sliding_mode_ocp_description()
        u1_est = np.zeros((opts.N_stages, 1))
        u1_est[-1] = -2.5
        u1_est[-2] = 0.1
        u2_est = np.zeros((opts.N_stages, 1))
        u2_est[-1] = 0.5
        u2_est[-2] = -2.0
        u_guess = np.concatenate((u1_est, u2_est), axis=1)
        X0 = X0.reshape((1, -1))
        x_est = np.concatenate((X0 * 2 / 3, X0 * 1 / 3, np.zeros((opts.N_stages - 2, 4))))

        opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
        solver = nosnoc.NosnocSolver(opts, model, ocp)
        solver.set("x", x_est)
        for i in range(opts.N_stages):
            self.assertTrue(
                np.array_equal(solver.problem.w0[solver.problem.ind_x[i][0][0]], x_est[i, :]))

        solver.set("u", u_guess)
        self.assertTrue(np.array_equal(solver.problem.w0[solver.problem.ind_u], u_guess))

        print("Solve with good initialization")
        res_initialized = solver.solve()

        print("Solve with bad initialization")
        opts2.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
        solver2 = nosnoc.NosnocSolver(opts2, model2, ocp2)
        solver2.set("u", -u_guess)
        solver2.set("x", np.repeat(np.array([53, 72, 28, 36]), 36, 0).reshape((36,4)))
        res_bad_init = solver2.solve()
        ipopt_iter_initialized = sum(res_initialized["nlp_iter"])
        ipopt_iter_bad_init = sum(res_bad_init["nlp_iter"])

        print(f"{ipopt_iter_initialized=} \t {ipopt_iter_bad_init=}")

        # If initialized, the iteration should be faster
        self.assertLessEqual(ipopt_iter_initialized, ipopt_iter_bad_init)


if __name__ == "__main__":
    unittest.main()
