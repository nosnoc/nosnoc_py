from examples.oscillator.oscillator_example import (
    get_default_options,
    TSIM,
    X_SOL,
    solve_oscillator,
)
from parameterized import parameterized
import unittest
import nosnoc
import numpy as np

EXACT_SWITCH_TIME = 1
X_SWITCH_EXACT = np.array([1.0, 0.0])

options = [
    (rk_representation, rk_scheme, dcs_mode)
    for dcs_mode in nosnoc.DcsMode
    for rk_scheme in nosnoc.RKScheme
    for rk_representation in nosnoc.RKRepresentation
]


def compute_errors(integrator) -> dict:
    X_sim = integrator.get("x")
    t_grid = integrator.get_time_grid()
    switch_diff = np.abs(t_grid - EXACT_SWITCH_TIME)
    err_t_switch = np.min(switch_diff)

    switch_index = np.where(switch_diff == err_t_switch)[0][0]
    err_x_switch = np.max(np.abs(X_sim[switch_index] - X_SWITCH_EXACT))

    err_t_end = np.abs(t_grid[-1] - TSIM)

    err_x_end = np.max(np.abs(X_sim[-1,:] - X_SOL))
    return {
        "t_switch": err_t_switch,
        "t_end": err_t_end,
        "x_switch": err_x_switch,
        "x_end": err_x_end,
    }


class OscillatorTests(unittest.TestCase):

    @parameterized.expand(options)
    def test_oscillator(self, rk_representation, rk_scheme, dcs_mode):
        opts = get_default_options(
            rk_representation=rk_representation,
            rk_scheme=rk_scheme,
            dcs_mode=dcs_mode,
        )
        opts.print_level = 0
        integrator = solve_oscillator(opts, do_plot=False)
        errors = compute_errors(integrator)

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
