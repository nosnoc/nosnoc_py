# This is an example use of path complementarities to enforce not braking and accelerating
# at the same time.

import nosnoc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([0, 0])
X_TARGET = np.array([500, 0])

def car_model():
    q = ca.SX.sym('q')
    v = ca.SX.sym('v')
    x = ca.vertcat(q, v)

    u = ca.SX.sym('u',2)
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
        [0,0],
        [k1,-kb]
        ])
    B2 = np.array([
        [0,0],
        [k2,-kb]
        ])

    f_1 = A@x + B1@u
    f_2 = A@x + B2@u

    F = [ca.horzcat(f_1, f_2)]

    c = [v-15]
    S = [np.array([[-1],[1]])]

    g_terminal = x - X_TARGET

    f_q = j_a*u[0]**2 + j_b*u[1]**2

    g_path_comp = ca.horzcat(u[0], u[1])

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, u=u)
    ocp = nosnoc.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, g_terminal=g_terminal, g_path_comp=g_path_comp)

    return model, ocp


def get_default_options():
    opts = nosnoc.NosnocOpts()
    # opts.pss_mode = nosnoc.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-6
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
    opts.print_level = 1

    opts.N_stages = 30
    opts.N_finite_elements = 2
    return opts

def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    model, ocp = car_model()
    opts.terminal_time = 30

    solver = nosnoc.NosnocSolver(opts, model, ocp)

    results = solver.solve()

    return results

def plot_car_model(results, latexify=True):
    x_traj = np.array(results['x_traj'])
    u_traj = np.array(results['u_traj'])
    t_grid = results['t_grid']
    t_grid_u = results['t_grid_u']

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
    plt.step(t_grid_u, np.concatenate([[u_traj[0,0]], u_traj[:, 0]]))
    plt.ylabel("$u_a$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.step(t_grid_u, np.concatenate([[u_traj[0,1]], u_traj[:, 1]]))
    plt.ylabel("$u_b$")
    plt.xlabel("time [s]")
    plt.grid()

    plt.show()

def example():
    results = solve_ocp()
    plot_car_model(results)

if __name__ == "__main__":
    example()
