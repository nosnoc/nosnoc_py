# This is an example use of path complementarities to enforce not braking and accelerating
# at the same time.

import nosnoc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([0, 0])
X_TARGET = np.array([500, 0])

TERMINAL_TIME = 30

def car_model():
    q = ca.SX.sym('q')
    v = ca.SX.sym('v')
    x = ca.vertcat(q, v)

    u = ca.SX.sym('u', 2)
    lbu = np.zeros((2,))
    ubu = np.ones((2,))

    k1 = 5
    k2 = 3
    kb = 5

    j_a = 1
    j_b = 1

    A = np.array([
        [0, 1],
        [0, 0]
    ])

    B1 = np.array([
        [0, 0],
        [k1, -kb]
        ])
    B2 = np.array([
        [0, 0],
        [k2, -kb]
        ])

    f_1 = A@x + B1@u
    f_2 = A@x + B2@u

    F = [ca.horzcat(f_1, f_2)]

    c = [v-15]
    S = [np.array([[-1], [1]])]

    g_terminal = x - X_TARGET

    f_q = j_a*u[0]**2 + j_b*u[1]**2

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=X0, u=u, lbu=lbu, ubu=ubu, f_q=f_q, g_terminal=g_terminal, G_path=u[0], H_path=u[1])

    return model

def get_default_options(**kwargs) -> nosnoc.Options:
    default_args = {
        "N_stages":30,
        "N_finite_elements":3,
        "n_s":2,
        "T":TERMINAL_TIME,
        "use_fesd":True,
        "cross_comp_mode":nosnoc.CrossComplementarityMode.FE_FE,
        "rho_h": 10.0
        }
    merged = dict(list(default_args.items())+ list(kwargs.items()))
    opts = nosnoc.Options(
        **merged
    )
    return opts


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    model = car_model()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    solver.solve()

    return solver


def plot_car_model(solver, latexify=True):
    x_traj = solver.get("x")
    u_traj = solver.get("u")
    t_grid = solver.get_time_grid()
    t_grid_u = solver.get_control_grid()

    if latexify:
        nosnoc.latexify_plot()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_grid, x_traj[:, 0])
    plt.ylabel("$x$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t_grid, x_traj[:, 1])
    plt.ylabel("$v$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t_grid_u, np.concatenate([[u_traj[0, 0]], u_traj[:, 0]]))
    plt.ylabel("$u_a$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.step(t_grid_u, np.concatenate([[u_traj[0, 1]], u_traj[:, 1]]))
    plt.ylabel("$u_b$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.show()


def example(plot=True):
    solver = solve_ocp()
    if plot:
        plot_car_model(solver)


if __name__ == "__main__":
    example()
