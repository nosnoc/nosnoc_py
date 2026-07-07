import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, horzcat

import nosnoc
from nosnoc.nosnoc_types import CrossComplementarityMode, ConstraintRelaxationMode, StepEquilibrationMode

# example opts
TERMINAL_RELAXATION = ConstraintRelaxationMode.NONE
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


# solver opts
def get_default_options() -> nosnoc.Options:
    N_stages = 6
    N_fe = 6
    n_s = 2
    opts = nosnoc.Options(
        N_stages=N_stages,
        N_finite_elements=[N_fe]*N_stages,
        T=TERMINAL_TIME,
        use_fesd=True,
        cross_comp_mode=CrossComplementarityMode.FE_FE,
        #step_equilibration=StepEquilibrationMode.LINEAR_COMPLEMENTARITY,
        relax_terminal_constraint = TERMINAL_RELAXATION,
        n_s=n_s,
    )
    return opts


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

    g_terminal = x[:2] - X_TARGET
    f_terminal = SX.zeros(1)

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=X0, u=u, lbu=LBU, ubu=UBU, f_q=f_q, f_q_T=f_terminal, g_terminal=g_terminal)

    return model


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    model = get_sliding_mode_ocp_description()

    solver = nosnoc.OcpSolver(model, opts, solver_opts)
    solver.solve()
    breakpoint()
    return solver


def example(plot=True):
    solver = solve_ocp()
    if plot:
        plot_sliding_mode(
            solver.get("x"),
            solver.get("u"),
            solver.get_time_grid(),
            solver.get_control_grid(),
        )
        plot_time_steps(solver.get("h"))


def plot_sliding_mode(x_traj, u_traj, t_grid, t_grid_u, latexify=True):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t_grid_u, np.vstack([u_traj[0,:], u_traj]), label="u")
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
