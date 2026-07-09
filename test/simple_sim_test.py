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
import nosnoc
import numpy as np

NS_VALUES = range(1, 4)
N_FINITE_ELEMENT_VALUES = range(2, 4)

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

    def test_default(self):
        model = get_simplest_model_sliding()
        check_opts(get_default_options(), model)

    def test_switch(self):
        model = get_simplest_model_switch()

        for ns in NS_VALUES:
            for Nfe in N_FINITE_ELEMENT_VALUES:
                for dcs_mode in [nosnoc.DcsMode.STEWART, nosnoc.DcsMode.STEP]:
                    for cross_comp_mode in nosnoc.CrossComplementarityMode:
                        for rk_scheme in nosnoc.RKScheme:
                            opts = get_default_options(
                                step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_DELTA,
                                n_s = ns,
                                N_finite_elements = Nfe,
                                dcs_mode = dcs_mode,
                                cross_comp_mode = cross_comp_mode,
                                print_level = 0,
                                rk_scheme = rk_scheme,
                            )
                        try:
                            check_opts(opts, model=model)
                        except:
                            raise Exception(f"test_switch failed with setting:\n {ns=} {Nfe=} {dcs_mode=} {cross_comp_mode=}")
        print("main_test_switch: SUCCESS")

    def test_sliding(self):
        model = get_simplest_model_sliding()

        for ns in NS_VALUES:
            for Nfe in N_FINITE_ELEMENT_VALUES:
                for dcs_mode in [nosnoc.DcsMode.STEWART, nosnoc.DcsMode.STEP]:
                    for rk_scheme in nosnoc.RKScheme:
                        opts = get_default_options(
                            step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,
                            rk_scheme = rk_scheme,
                            print_level = 0,
                            n_s = ns,
                            N_finite_elements = Nfe,
                            dcs_mode = dcs_mode,
                        )
                        try:
                            check_opts(opts, model=model)
                        except:
                            raise Exception(f"test_sliding failed with setting:\n {ns=} {Nfe=} {dcs_mode=} {rk_scheme=}")
        print("main_test_sliding: SUCCESS")

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

    def test_discretization(self):
        model = get_simplest_model_sliding()

        for rk_scheme in nosnoc.RKScheme:
            for rk_representation in nosnoc.RKRepresentation:
                for dcs_mode in [nosnoc.DcsMode.STEWART, nosnoc.DcsMode.STEP]:
                    for cross_comp_mode in nosnoc.CrossComplementarityMode:
                        opts = get_default_options(
                            rk_scheme = rk_scheme,
                            rk_representation = rk_representation,
                            dcs_mode = dcs_mode,
                            cross_comp_mode = cross_comp_mode,
                            print_level = 0,
                        )
                        try:
                            check_opts(opts, model=model)
                        except:
                            raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")
        print("main_test_sliding: SUCCESS")

if __name__ == "__main__":
    unittest.main()
    # uncomment to run single test locally
    # simple_test = SimpleTests()
    # simple_test.test_least_squares_problem()
