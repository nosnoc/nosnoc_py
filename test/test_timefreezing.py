import unittest
from examples.temperature_control.time_freezing_hysteresis_temperature_control import control


class TestTimeFreezing(unittest.TestCase):

    def test_control_example(self):
        model, opts, solver, results = control(with_plot=False)
        end_time = model.t_fun(results["x_list"][-1])
        self.assertLessEqual(opts.terminal_time - opts.time_freezing_tolerance, end_time)
        self.assertLessEqual(end_time, opts.terminal_time + opts.time_freezing_tolerance)


if __name__ == "__main__":
    unittest.main()
