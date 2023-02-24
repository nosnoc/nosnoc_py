from examples.oscilator.oscilator_example import (
    TSIM,
    get_oscilator_model,
    plot_oscilator,
)
import unittest
import nosnoc
import numpy as np


X0 = np.array([9.6325807390e-01, -1.5270760151e-01])
XEXACT = np.array([1.0123595211e+00, -1.8747024319e-01])
TSTEP = TSIM / 29

class OscilatorSwitchTest(unittest.TestCase):
    def test_default_solver(self, opts=None, use_g_Stewart=False, do_plot=False):
        opts = nosnoc.NosnocOpts()
        opts.terminal_time = TSTEP

        opts.print_level = 1
        # opts.opts_casadi_nlp['ipopt']['print_level'] = 5
        # opts.opts_casadi_nlp['print_time'] = 1

        # opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
        opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_EQ
        # opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER
        # opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
        # opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT_COMPLEMENTARITY
        opts.initialization_strategy = nosnoc.InitializationStrategy.ALL_XCURRENT_W0_START
        opts.initialization_strategy = nosnoc.InitializationStrategy.RK4_SMOOTHENED
        opts.sigma_0 = 1e0
        # opts.fb_ip_aug1_weight = 1e-1  # ball
        # opts.fb_ip_aug2_weight = 1e-0  # banana

        model = get_oscilator_model(use_g_Stewart)
        model.x0 = X0

        solver = nosnoc.NosnocSolver(opts, model)
        results = solver.solve()

        x_out = results["x_all_list"][-1]
        error = np.max(np.abs(XEXACT - x_out))
        print(f"error wrt exact solution {error:.2e}")
        print(f"x_out {x_out[0]}, {x_out[1]}")

        if do_plot:
            plot_oscilator([X0] + results["x_list"], results["t_grid"])
        assert(error < 1e-4)



    def test_custom_solver(self, opts=None, use_g_Stewart=False, do_plot=False):
        opts = nosnoc.NosnocOpts()
        # opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT_COMPLEMENTARITY
        opts.terminal_time = TSTEP
        opts.sigma_N = 1e-5

        opts.print_level = 2

        opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
        opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ

        model = get_oscilator_model(use_g_Stewart)
        model.x0 = X0

        solver = nosnoc.NosnocCustomSolver(opts, model)
        results = solver.solve()

        x_out = results["x_all_list"][-1]
        error = np.max(np.abs(XEXACT - x_out))
        print(f"error wrt exact solution {error:.2e}")
        print(f"x_out {x_out[0]}, {x_out[1]}")

        if do_plot:
            plot_oscilator([X0] + results["x_list"], results["t_grid"])

        assert(error < 1e-4)


if __name__ == "__main__":
    unittest.main()
    # oscilator_test = OscilatorSwitchTest()
    # oscilator_test.solve_oscilator_fast(do_plot=True)
