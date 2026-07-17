from examples.oscillator.oscillator_example import (
    get_default_options,
    TSIM,
    X_SOL,
    solve_oscillator,
)
import unittest
import nosnoc
import numpy as np

EXACT_SWITCH_TIME = 1
X_SWITCH_EXACT = np.array([1.0, 0.0])


def compute_errors(results) -> dict:
    X_sim = results["X_sim"]
    switch_diff = np.abs(results["t_grid"] - EXACT_SWITCH_TIME)
    err_t_switch = np.min(switch_diff)

    switch_index = np.where(switch_diff == err_t_switch)[0][0]
    err_x_switch = np.max(np.abs(X_sim[switch_index] - X_SWITCH_EXACT))

    err_t_end = np.abs(results["t_grid"][-1] - TSIM)

    err_x_end = np.max(np.abs(X_sim[-1] - X_SOL))
    return {
        "t_switch": err_t_switch,
        "t_end": err_t_end,
        "x_switch": err_x_switch,
        "x_end": err_x_end,
    }


class OscillatorTests(unittest.TestCase):

    def test_default(self):
        opts = get_default_options()
        opts.print_level = 0
        results = solve_oscillator(opts, do_plot=False)
        errors = compute_errors(results)

        print(errors)
        tol = 1e-5
        assert errors["t_switch"] < tol
        assert errors["t_end"] < tol
        assert errors["x_switch"] < tol
        assert errors["x_end"] < tol

    def test_polishing(self):
        opts = get_default_options()
        opts.print_level = 0
        opts.comp_tol = 1e-4
        opts.do_polishing_step = True

        opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
        opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
        # opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
        # opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER

        results = solve_oscillator(opts, do_plot=False)
        errors = compute_errors(results)

        print(errors)
        tol = 1e-5
        assert errors["t_switch"] < tol
        assert errors["t_end"] < tol
        assert errors["x_switch"] < tol
        assert errors["x_end"] < tol


    def test_fix_active_set(self):
        opts = get_default_options()
        opts.print_level = 0
        opts.fix_active_set_fe0 = True

        opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
        opts.step_equilibration = nosnoc.StepEquilibrationMode.L2_RELAXED

        results = solve_oscillator(opts, do_plot=False)
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
    # oscillator_test = OscillatorTests()
    # oscillator_test.test_least_squares_problem()
