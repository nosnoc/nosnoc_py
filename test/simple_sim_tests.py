from examples.simplest_example import (
    TOL,
    get_default_options,
    X0,
    TSIM,
    EXACT_SWITCH_TIME,
    solve_simplest_example,
    get_simplest_model_sliding,
    get_simplest_model_switch,
)
import nosnoc
import numpy as np

NS_VALUES = range(1, 4)
N_FINITE_ELEMENT_VALUES = range(2, 4)

NO_FESD_X_END = 0.36692644


def compute_errors(results, model) -> dict:
    X_sim = results["X_sim"]
    err_x0 = np.abs(X_sim[0] - X0)

    switch_diff = np.abs(results["t_grid"] - EXACT_SWITCH_TIME)
    err_t_switch = np.min(switch_diff)

    switch_index = np.where(switch_diff == err_t_switch)[0][0]
    err_x_switch = np.abs(X_sim[switch_index])

    err_t_end = np.abs(results["t_grid"][-1] - TSIM)

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


def test_opts(opts, model):
    results = solve_simplest_example(opts=opts, model=model)
    errors = compute_errors(results, model)

    print(errors)
    tol = 1e1 * TOL
    assert errors["x0"] < tol
    assert errors["t_switch"] < tol
    assert errors["t_end"] < tol
    assert errors["x_switch"] < tol
    assert errors["x_end"] < tol


def test_default():
    model = get_simplest_model_sliding()
    test_opts(get_default_options(), model)


def main_test_switch():
    model = get_simplest_model_switch()

    for ns in NS_VALUES:
        for Nfe in N_FINITE_ELEMENT_VALUES:
            for pss_mode in nosnoc.PssMode:
                for cross_comp_mode in nosnoc.CrossComplementarityMode:
                    opts = get_default_options()
                    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_DELTA
                    opts.print_level = 0
                    opts.n_s = ns
                    opts.N_finite_elements = Nfe
                    opts.pss_mode = pss_mode
                    opts.cross_comp_mode = cross_comp_mode
                    opts.print_level = 1
                    try:
                        test_opts(opts, model=model)
                    except:
                        raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")
    print("main_test_switch: SUCCESS")


def main_test_sliding():
    model = get_simplest_model_sliding()

    for ns in NS_VALUES:
        for Nfe in N_FINITE_ELEMENT_VALUES:
            for pss_mode in nosnoc.PssMode:
                for irk_scheme in nosnoc.IrkSchemes:
                    opts = get_default_options()
                    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
                    opts.irk_scheme = irk_scheme
                    opts.print_level = 0
                    opts.n_s = ns
                    opts.N_finite_elements = Nfe
                    opts.pss_mode = pss_mode
                    try:
                        test_opts(opts, model=model)
                    except:
                        raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")
    print("main_test_sliding: SUCCESS")


def main_test_discretization():
    model = get_simplest_model_sliding()

    for mpcc_mode in [nosnoc.MpccMode.SCHOLTES_EQ, nosnoc.MpccMode.SCHOLTES_INEQ]:
        for irk_scheme in nosnoc.IrkSchemes:
            for irk_representation in nosnoc.IrkRepresentation:
                opts = get_default_options()
                opts.mpcc_mode = mpcc_mode
                opts.irk_scheme = irk_scheme
                opts.print_level = 0
                opts.irk_representation = irk_representation
                try:
                    test_opts(opts, model=model)
                except:
                    raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")
    print("main_test_sliding: SUCCESS")


def main_test_fesd_off():
    model = get_simplest_model_switch()

    opts = get_default_options()
    opts.print_level = 0
    opts.use_fesd = False

    try:
        # solve
        results = solve_simplest_example(opts=opts, model=model)

        errors = compute_errors(results, model)
        tol = 1e1 * TOL
        # these should be off
        assert errors["x_end"] > 0.01
        assert errors["t_switch"] > 0.01
        # these should be correct
        assert errors["x0"] < tol
        assert errors["t_end"] < tol
        #
        assert np.allclose(results["time_steps"], np.min(results["time_steps"]))
    except:
        raise Exception("Test with FESD off failed")
    print("main_test_fesd_off: SUCCESS")


def main_test_least_squares_problem():
    model = get_simplest_model_switch()

    opts = get_default_options()
    opts.print_level = 1
    opts.n_s = 3
    opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
    opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
    opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER
    opts.irk_scheme = nosnoc.IrkSchemes.RADAU_IIA
    opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
    opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
    opts.sigma_0 = 0.1
    try:
        test_opts(opts, model=model)
    except:
        raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")

if __name__ == "__main__":
    test_default()
    main_test_fesd_off()
    main_test_discretization()
    main_test_sliding()
    main_test_switch()
    main_test_least_squares_problem()
