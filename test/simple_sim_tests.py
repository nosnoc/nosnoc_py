from examples.simplest.simplest_example import (
    TOL,
    get_default_options,
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
    return results


class SimpleTests(unittest.TestCase):

    def test_default(self):
        model = get_simplest_model_sliding()
        test_opts(get_default_options(), model)

    def test_switch(self):
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
                        opts.print_level = 0
                        try:
                            test_opts(opts, model=model)
                        except:
                            raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")
        print("main_test_switch: SUCCESS")

    def test_sliding(self):
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

    def test_discretization(self):
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

    def test_fesd_off(self):
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

    def test_least_squares_problem(self):
        model = get_simplest_model_switch()

        opts = get_default_options()
        opts.print_level = 2
        opts.n_s = 2

        opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
        opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER

        opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER_IP_AUG
        # opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER

        # opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
        opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT_COMPLEMENTARITY

        opts.initialization_strategy = nosnoc.InitializationStrategy.ALL_XCURRENT_W0_START
        # opts.fix_active_set_fe0 = True
        opts.sigma_0 = 1e0
        opts.gamma_h = np.inf

        # opts.comp_tol = 1e-5
        # opts.do_polishing_step = True
        try:
            results = test_opts(opts, model=model)
            print(results["t_grid"])
        except:
            raise Exception(f"Test failed.")


    def test_least_squares_problem_opts(self):
        model = get_simplest_model_switch()

        for step_equilibration in [nosnoc.StepEquilibrationMode.DIRECT_COMPLEMENTARITY, nosnoc.StepEquilibrationMode.DIRECT]:
            for fix_as in [True, False]:
                opts = get_default_options()
                opts.fix_active_set_fe0 = fix_as
                opts.print_level = 2

                opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
                opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER

                opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER_IP_AUG

                opts.step_equilibration = step_equilibration

                opts.initialization_strategy = nosnoc.InitializationStrategy.ALL_XCURRENT_W0_START
                opts.sigma_0 = 1e0
                opts.gamma_h = np.inf

                try:
                    results = test_opts(opts, model=model)
                    # print(results["t_grid"])
                except:
                    # print(f"Test failed with {fix_as=}, {step_equilibration=}")
                    raise Exception(f"Test failed with {fix_as=}, {step_equilibration=}")



    def test_initializations(self):
        model = get_simplest_model_switch()

        for initialization_strategy in nosnoc.InitializationStrategy:
            opts = get_default_options()
            opts.print_level = 0
            opts.initialization_strategy = initialization_strategy
            print(f"\ntesting initialization_strategy = {initialization_strategy}")
            try:
                test_opts(opts, model=model)
            except:
                raise Exception(f"Test failed with setting:\n {opts=}")

    def test_polishing(self):
        model = get_simplest_model_switch()

        opts = get_default_options()
        opts.print_level = 1
        opts.do_polishing_step = True
        opts.comp_tol = 1e-3
        try:
            test_opts(opts, model=model)
        except:
            raise Exception(f"Test failed with setting:\n {opts=} \n{model=}")


if __name__ == "__main__":
    unittest.main()
    # uncomment to run single test locally
    # simple_test = SimpleTests()
    # simple_test.test_least_squares_problem()
