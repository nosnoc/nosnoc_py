from examples.relay.relay_feedback_system import (
    get_relay_feedback_system_model,
    get_default_options
)
import unittest
import nosnoc
import numpy as np

class CustomSolverRelayTests(unittest.TestCase):

    def test_oscilator_sim(self):
        opts = get_default_options()
        Tsim = 10
        Nsim = 200
        Tstep = Tsim / Nsim

        opts.terminal_time = Tstep
        opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
        # opts.print_level = 1

        model = get_relay_feedback_system_model()
        solver = nosnoc.NosnocCustomSolver(opts, model)

        # simulation loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
        looper.run()
        results = looper.get_results()
        assert all([status == nosnoc.Status.SUCCESS for status in results["status"]])

        x_ref = np.array([-2.77699355e-06, -8.82387155e-01,  2.37411855e+00])
        x_terminal = results["X_sim"][-1]
        # relative error:
        error = np.max(np.abs(x_ref - x_terminal)) / np.max(np.abs(x_ref))
        assert np.max(error) < 1e-4

if __name__ == "__main__":
    unittest.main()