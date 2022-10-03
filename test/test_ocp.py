import numpy as np
import nosnoc
from examples.sliding_mode_ocp import (
    solve_ocp,
    example,
    get_default_settings,
    X0,
    X_TARGET,
    TERMINAL_TIME,
    UBU,
    LBU,
)

EQUIDISTANT_CONTROLS = [True, False]
PSS_MODES = [nosnoc.PssMode.STEWART]


def test_default():
    example(plot=False)


def test_loop():
    for equidistant_control_grid in EQUIDISTANT_CONTROLS:
        for step_equilibration in nosnoc.StepEquilibrationMode:
            for irk_representation in nosnoc.IrkRepresentation:
                for irk_scheme in nosnoc.IRKSchemes:
                    for pss_mode in PSS_MODES:
                        settings = get_default_settings()
                        settings.comp_tol = 1e-5
                        settings.N_stages = 5
                        settings.N_finite_elements = 2
                        settings.equidistant_control_grid = equidistant_control_grid
                        settings.step_equilibration = step_equilibration
                        settings.irk_representation = irk_representation
                        settings.irk_scheme = irk_scheme
                        settings.pss_mode = pss_mode

                        print(f"test setting: {settings}")
                        results = solve_ocp(settings)

                        x_traj = results["x_traj"]
                        u_traj = results["u_traj"]
                        t_grid = results["t_grid"]

                        np.allclose(x_traj[0], X0, atol=1e-4)
                        np.allclose(x_traj[-1][:2], X_TARGET, atol=1e-4)
                        np.allclose(t_grid[-1], TERMINAL_TIME, atol=1e-6)
                        np.allclose(t_grid[0], 0.0, atol=1e-6)
                        np.alltrue(u_traj < UBU)
                        np.alltrue(u_traj > LBU)


if __name__ == "__main__":
    test_default()
    test_loop()
