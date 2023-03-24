from examples.oscillator.oscillator_example import (
    get_default_options,
    TSIM,
    X_SOL,
    get_oscillator_model,
)
from oscillator_test import compute_errors
import unittest
import nosnoc
import numpy as np

EXACT_SWITCH_TIME = 1
X_SWITCH_EXACT = np.array([1.0, 0.0])

class CustomSolverOscilatorTests(unittest.TestCase):

    def test_oscillator_sim(self):
        opts = get_default_options()
        opts.print_level = 1
        opts.n_s = 3
        opts.N_finite_elements = 3

        opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
        opts.sigma_0 = 1e0

        model = get_oscillator_model()
        Nsim = 29
        Tstep = TSIM / Nsim
        opts.terminal_time = Tstep

        solver = nosnoc.NosnocCustomSolver(opts, model)

        # loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
        looper.run()
        results = looper.get_results()

        error = np.max(np.abs(X_SOL - results["X_sim"][-1]))
        print(f"error wrt exact solution {error:.2e}")
        # check all results['status'] are nosnoc.Status.SUCCESS
        assert all([status == nosnoc.Status.SUCCESS for status in results["status"]])

        errors = compute_errors(results)

        print(errors)
        tol = 1e-5
        assert errors["t_switch"] < tol
        assert errors["t_end"] < tol
        assert errors["x_switch"] < tol
        assert errors["x_end"] < tol

if __name__ == "__main__":
    unittest.main()
    # uncomment to run single test locally
    # oscillator_test = CustomSolverTests()
    # oscillator_test.test_oscillator_sim()