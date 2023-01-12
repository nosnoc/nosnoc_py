from examples.motor_with_friction_ocp import (
    solve_ocp,
    example,
    get_default_options,
    X0,
    X_TARGET,
)
import nosnoc
import numpy as np


def test_default():
    example(plot=False)


def test_loop():

    for step_equilibration in nosnoc.StepEquilibrationMode:
        for pss_mode in nosnoc.PssMode:
            if step_equilibration != nosnoc.StepEquilibrationMode.DIRECT:
                opts = get_default_options()
                opts.step_equilibration = step_equilibration
                opts.pss_mode = pss_mode
                # opts.print_level = 0

                # print(f"test setting: {opts}")
                results = solve_ocp(opts)

                x_traj = results["x_traj"]

                assert np.allclose(x_traj[0], X0, atol=1e-4)
                assert np.allclose(x_traj[-1], X_TARGET, atol=1e-4)


if __name__ == "__main__":
    test_loop()
    test_default()
