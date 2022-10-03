import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, horzcat

import nosnoc

# example settings
TERMINAL_CONSTRAINT = True
LINEAR_CONTROL = True
TERMINAL_TIME = 4.0

if LINEAR_CONTROL:
    U_MAX = 10
    V0 = np.zeros((2,))
else:
    U_MAX = 2
    V0 = np.zeros((0,))
X0 = np.concatenate((np.array([2 * np.pi / 3, np.pi / 3]), V0))

X_TARGET = np.array([-np.pi / 6, -np.pi / 4])

# constraints
LBU = -U_MAX * np.ones((2,))
UBU = U_MAX * np.ones((2,))


# solver settings
def get_default_settings() -> nosnoc.NosnocSettings:
    settings = nosnoc.NosnocSettings()
    settings.irk_representation = nosnoc.IrkRepresentation.DIFFERENTIAL
    settings.comp_tol = 1e-9
    settings.homotopy_update_slope = 0.1
    settings.n_s = 2
    settings.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    settings.rho_h = 1e1
    settings.print_level = 1
    settings.N_stages = 6
    settings.N_finite_elements = 6
    return settings


def get_sliding_mode_ocp_description():

    # Variable defintion
    x1 = SX.sym("x1")
    x2 = SX.sym("x2")

    v1 = SX.sym("v1")
    v2 = SX.sym("v2")

    # Control
    u1 = SX.sym("u1")
    u2 = SX.sym("u2")
    u = vertcat(u1, u2)

    if LINEAR_CONTROL:
        x = vertcat(x1, x2, v1, v2)

        # dynamics
        f_11 = vertcat(-1 + v1, 0, u1, u2)
        f_12 = vertcat(1 + v1, 0, u1, u2)
        f_21 = vertcat(0, -1 + v2, u1, u2)
        f_22 = vertcat(0, 1 + v2, u1, u2)

        # Objective
        f_q = v1**2 + v2**2
    else:
        x = vertcat(x1, x2)

        # dynamics
        f_11 = vertcat(-1 + u1, 0)
        f_12 = vertcat(1 + u1, 0)
        f_21 = vertcat(0, -1 + u2)
        f_22 = vertcat(0, 1 + u2)

        # Objective
        f_q = u1**2 + u2**2

    # Switching Functions
    p = 2
    a = 0.15
    a1 = 0
    b = -0.05
    q = 3

    c1 = x1 + a * (x2 - a1)**p
    c2 = x2 + b * x1**q
    c = [c1, c2]
    S1 = np.array([[1], [-1]])
    S2 = np.array([[1], [-1]])
    S = [S1, S2]

    # Modes of the ODEs layers
    F1 = horzcat(f_11, f_12)
    F2 = horzcat(f_21, f_22)
    F = [F1, F2]

    if TERMINAL_CONSTRAINT:
        g_terminal = x[:2] - X_TARGET
        f_q_T = SX.zeros(1)
    else:
        g_terminal = SX.zeros(0)
        f_q_T = 100 * (x[:2] - X_TARGET).T @ (x[:2] - X_TARGET)

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, u=u)
    ocp = nosnoc.NosnocOcp(lbu=LBU, ubu=UBU, f_q=f_q, f_q_T=f_q_T, g_terminal=g_terminal)

    return model, ocp


def solve_ocp(settings=None):
    if settings is None:
        settings = get_default_settings()

    [model, ocp] = get_sliding_mode_ocp_description()

    settings.terminal_time = TERMINAL_TIME

    solver = nosnoc.NosnocSolver(settings, model, ocp)

    results = solver.solve()

    return results


def example(plot=True):
    results = solve_ocp()
    if plot:
        plot_sliding_mode(
            results["x_traj"],
            results["u_traj"],
            results["t_grid"],
            results["t_grid_u"],
        )
        plot_time_steps(results["time_steps"])


def plot_sliding_mode(x_traj, u_traj, t_grid, t_grid_u, latexify=True):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t_grid_u, [u_traj[0]] + u_traj, label="u")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_grid, x_traj, label="x")
    plt.legend()
    plt.grid()

    plt.show()


def plot_time_steps(t_steps):
    n = len(t_steps)
    plt.figure()
    plt.step(list(range(n)), t_steps[0] + t_steps)
    plt.grid()
    plt.ylabel("time_step [s]")
    plt.ylabel("time_step index")
    plt.show()


if __name__ == "__main__":
    example()
