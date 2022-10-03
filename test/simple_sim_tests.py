from examples.simplest_example import (
    TOL,
    get_default_settings,
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


def test_settings(settings, model):
    results = solve_simplest_example(settings=settings, model=model)
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
    test_settings(get_default_settings(), model)


def main_test_switch():
    model = get_simplest_model_switch()

    for ns in NS_VALUES:
        for Nfe in N_FINITE_ELEMENT_VALUES:
            for pss_mode in nosnoc.PssMode:
                for cross_comp_mode in nosnoc.CrossComplementarityMode:
                    settings = get_default_settings()
                    settings.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_DELTA
                    settings.print_level = 0
                    settings.n_s = ns
                    settings.N_finite_elements = Nfe
                    settings.pss_mode = pss_mode
                    settings.cross_comp_mode = cross_comp_mode
                    settings.print_level = 1
                    try:
                        test_settings(settings, model=model)
                    except:
                        raise Exception(f"Test failed with setting:\n {settings=} \n{model=}")
    print("main_test_switch: SUCCESS")


def main_test_sliding():
    model = get_simplest_model_sliding()

    for ns in NS_VALUES:
        for Nfe in N_FINITE_ELEMENT_VALUES:
            for pss_mode in nosnoc.PssMode:
                for irk_scheme in nosnoc.IRKSchemes:
                    settings = get_default_settings()
                    settings.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
                    settings.irk_scheme = irk_scheme
                    settings.print_level = 0
                    settings.n_s = ns
                    settings.N_finite_elements = Nfe
                    settings.pss_mode = pss_mode
                    try:
                        test_settings(settings, model=model)
                    except:
                        raise Exception(f"Test failed with setting:\n {settings=} \n{model=}")
    print("main_test_sliding: SUCCESS")


def main_test_discretization():
    model = get_simplest_model_sliding()

    for mpcc_mode in nosnoc.MpccMode:
        for irk_scheme in nosnoc.IRKSchemes:
            for irk_representation in nosnoc.IrkRepresentation:
                for lifted_irk in [True, False]:
                    settings = get_default_settings()
                    settings.mpcc_mode = mpcc_mode
                    settings.irk_scheme = irk_scheme
                    settings.print_level = 0
                    settings.irk_representation = irk_representation
                    settings.lift_irk_differential = lifted_irk
                    try:
                        test_settings(settings, model=model)
                    except:
                        raise Exception(f"Test failed with setting:\n {settings=} \n{model=}")
    print("main_test_sliding: SUCCESS")


def main_test_fesd_off():
    model = get_simplest_model_switch()

    settings = get_default_settings()
    settings.print_level = 0
    settings.use_fesd = False

    try:
        # solve
        results = solve_simplest_example(settings=settings, model=model)

        X_sim = results["X_sim"]
        err_x0 = np.abs(X_sim[0] - X0)

        switch_diff = np.abs(results["t_grid"] - TSIM / 2)
        err_t_switch = np.min(switch_diff)

        err_t_end = np.abs(results["t_grid"][-1] - TSIM)
        err_x_end = np.abs(X_sim[-1] - NO_FESD_X_END)
        tol = 1e1 * TOL
        assert err_x_end < tol
        assert err_t_end < tol
        assert err_t_switch < tol
    except:
        raise Exception("Test with FESD off failed")
    print("main_test_fesd_off: SUCCESS")


if __name__ == "__main__":
    test_default()
    main_test_fesd_off()
    main_test_discretization()
    main_test_sliding()
    main_test_switch()
