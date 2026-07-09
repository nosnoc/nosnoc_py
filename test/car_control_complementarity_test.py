import unittest
from parameterized import parameterized
import numpy as np
import nosnoc
from examples.car_control_complementarity.car_control_complementarity import (
    solve_ocp,
    example,
    get_default_options,
    X0,
    X_TARGET,
    TERMINAL_TIME,
)

TIME_OPTIMAL = [True, False]
USE_SPEED_OF_TIME = [True, False]
LOCAL_SPEED_OF_TIME = [True, False]
STEP_EQUILIBRATION_MODES = [
    nosnoc.StepEquilibrationMode.HEURISTIC_MEAN, nosnoc.StepEquilibrationMode.HEURISTIC_DELTA,
    nosnoc.StepEquilibrationMode.L2_RELAXED, nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
]

options = [
    (time_optimal, use_speed_of_time, local_speed_of_time, rk_representation, rk_scheme, dcs_mode)
    for time_optimal in TIME_OPTIMAL
    for use_speed_of_time in USE_SPEED_OF_TIME
    for local_speed_of_time in LOCAL_SPEED_OF_TIME
    for rk_representation in nosnoc.RKRepresentation
    for rk_scheme in nosnoc.RKScheme
    for dcs_mode in nosnoc.DcsMode
    if time_optimal or (not use_speed_of_time and not local_speed_of_time)
]


class TestCarControlComplementarity(unittest.TestCase):

    def test_default(self):
        example(plot=False)

    @parameterized.expand(options)
    def test_combination(self, time_optimal, use_speed_of_time, local_speed_of_time, rk_representation, rk_scheme, dcs_mode):
        opts = get_default_options(
            time_optimal_problem=time_optimal,
            use_speed_of_time_variables=use_speed_of_time,
            local_speed_of_time_variable=local_speed_of_time,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme,
            dcs_mode = dcs_mode,
        )
        message = (
            f"""
            Test setting:
            time_optimal: {time_optimal}
            use_speed_of_time: {use_speed_of_time}
            local_speed_of_time: {local_speed_of_time}
            rk_representation: {rk_representation}
            rk_scheme: {rk_scheme}
            dcs_mode: {dcs_mode}
            """
        )
        print(message)
        solver = solve_ocp(opts)

        x_traj = solver.get("x")
        u_traj = solver.get("u")
        t_grid = solver.get_time_grid()
        x_tol = 1e-6
        u_tol = 1e-6

        print(f"t_grid = {t_grid}")
        print(f"x_traj[-1,:2] = {x_traj[-1,:2]}, X_TARGET = {X_TARGET}, error= {np.max(np.abs(x_traj[-1,:2] - X_TARGET))}")
        self.assertTrue(np.allclose(x_traj[0,:], X0, atol=1e-4), message)
        self.assertTrue(np.allclose(x_traj[-1,:2], X_TARGET, atol=x_tol), message)
        self.assertTrue(np.allclose(t_grid[0], 0.0, atol=1e-6), message)
        self.assertTrue(np.all(u_traj[:,0]*u_traj[:,1] < u_tol), message)
        self.assertTrue(np.all(u_traj >= 0), message)


if __name__ == "__main__":
    unittest.main()
