from examples.simplest.simplest_example import (
    get_default_options,
    get_default_integrator_options,
    X0,
    TSIM,
    EXACT_SWITCH_TIME,
    solve_simplest_example,
    get_simplest_model_sliding,
    get_simplest_model_switch,
)
import unittest
from parameterized import parameterized
import nosnoc
import numpy as np

NS_VALUES = range(1, 4)
N_FINITE_ELEMENT_VALUES = range(2, 4)

options = [
    (rk_representation, rk_scheme, dcs_mode, cross_comp_mode, step_eq_mode, N_fe, n_s)
    for N_fe in N_FINITE_ELEMENT_VALUES
    for n_s in NS_VALUES
    for cross_comp_mode in nosnoc.CrossComplementarityMode
    for step_eq_mode in nosnoc.StepEquilibrationMode
    for dcs_mode in nosnoc.DcsMode
    for rk_scheme in nosnoc.RKScheme
    for rk_representation in nosnoc.RKRepresentation
    if step_eq_mode is not nosnoc.StepEquilibrationMode.DIRECT_HOMOTOPY # Unimplemented
    if step_eq_mode is not nosnoc.StepEquilibrationMode.LINEAR_COMPLEMENTARITY # Extremely Flakey
    if step_eq_mode is not nosnoc.StepEquilibrationMode.DIRECT # Very Mildly Flakey
]


NO_FESD_X_END = 0.36692644

TOL = 1e-7

def compute_errors(integrator, model) -> dict:
    X_sim = integrator.get("x")
    t_grid = integrator.get_time_grid()
    err_x0 = np.abs(X_sim[0] - X0)

    switch_diff = np.abs(t_grid - EXACT_SWITCH_TIME)
    err_t_switch = np.min(switch_diff)

    switch_index = np.where(switch_diff == err_t_switch)[0][0]
    err_x_switch = np.abs(X_sim[switch_index])

    err_t_end = np.abs(t_grid[-1] - TSIM)

    x_end_ref = 0.0
    if "switch" in model.name:
        x_end_ref = TSIM - EXACT_SWITCH_TIME
    err_x_end = np.abs(X_sim[-1] - x_end_ref)
    return {
        "x0": err_x0,
        "t_switch": err_t_switch,
        "t_end": err_t_end,
        "x_switch": err_x_switch,
        "x_end": err_x_end,
    }


def check_opts(opts, model):
    _,_,_,_,integrator = solve_simplest_example(opts=opts, model=model)
    errors = compute_errors(integrator, model)

    print(errors)
    tol = 1e1 * TOL
    assert errors["x0"] < tol
    assert errors["t_switch"] < tol
    assert errors["t_end"] < tol
    assert errors["x_switch"] < tol
    assert errors["x_end"] < tol
    return integrator


class SimpleTests(unittest.TestCase):
    @parameterized.expand(options)
    def test_switch(self, rk_representation, rk_scheme, dcs_mode, cross_comp_mode, step_eq_mode, N_fe, n_s):
        model = get_simplest_model_switch()

        opts = get_default_options(
            step_equilibration = step_eq_mode,
            n_s = n_s,
            N_finite_elements = N_fe,
            dcs_mode = dcs_mode,
            cross_comp_mode = cross_comp_mode,
            print_level = 0,
            rk_scheme = rk_scheme,
            rk_representation = rk_representation,
        )
        check_opts(opts, model=model)

    @parameterized.expand(options)
    def test_sliding(self, rk_representation, rk_scheme, dcs_mode, cross_comp_mode, step_eq_mode, N_fe, n_s):
        model = get_simplest_model_sliding()

        opts = get_default_options(
            step_equilibration = step_eq_mode,
            n_s = n_s,
            N_finite_elements = N_fe,
            dcs_mode = dcs_mode,
            cross_comp_mode = cross_comp_mode,
            print_level = 0,
            rk_scheme = rk_scheme,
            rk_representation = rk_representation,
        )
        check_opts(opts, model=model)

    def test_fesd_off(self):
        model = get_simplest_model_switch()

        opts = get_default_options(
            print_level = 0,
            use_fesd = False,
        )

        try:
            # solve
            _,_,_,_,integrator = solve_simplest_example(opts=opts, model=model)

            errors = compute_errors(integrator, model)
            tol = 1e1 * TOL
            # these should be off
            assert errors["x_end"] > 0.01
            assert errors["t_switch"] > 0.01
            # these should be correct
            assert errors["x0"] < tol
            assert errors["t_end"] < tol
            #
        except:
            raise Exception("Test with FESD off failed")
        print("main_test_fesd_off: SUCCESS")


if __name__ == "__main__":
    unittest.main()
