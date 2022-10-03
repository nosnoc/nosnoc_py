from examples.motor_with_friction_ocp import (
    solve_ocp,
    example,
    get_default_settings,
    X0,
    X_TARGET,
)
import nosnoc
import numpy as np

PSS_MODES = nosnoc.PssMode


def test_default():
    example(plot=False)


def test_loop():
    settings = get_default_settings()

    for step_equilibration in nosnoc.StepEquilibrationMode:
        for pss_mode in PSS_MODES:
            settings.step_equilibration = step_equilibration
            settings.pss_mode = pss_mode

            print(f"test setting: {settings}")
            results = solve_ocp(settings)

            x_traj = results["x_traj"]

            np.allclose(x_traj[0], X0, atol=1e-4)
            np.allclose(x_traj[-1], X_TARGET, atol=1e-4)


if __name__ == "__main__":
    test_default()
    test_loop()
