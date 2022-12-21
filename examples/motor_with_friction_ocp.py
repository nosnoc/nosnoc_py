import nosnoc
from casadi import SX, vertcat, horzcat
import numpy as np
import matplotlib.pyplot as plt

# example opts
illustrate_regions = True
TERMINAL_CONSTRAINT = True
LINEAR_CONTROL = True
X0 = np.array([0, 0, 0, 0, 0])
X_TARGET = np.array([0.01, 0, 0.01, 0, 0])


def get_motor_with_friction_ocp_description():

    # Parameters
    m1 = 1.03  # slide mass
    m2 = 0.56  # load mass
    k = 2.4e3  # spring constant N/m
    c_damping = 0.00  # damping
    u_max = 5  # voltage Back-EMF, U = K_s*v_1
    R = 2  # coil resistance ohm
    L = 2e-3  # inductivity, henry
    K_F = 12  # force constant N/A, F_L = K_F*I # Lorenz force
    K_S = 12  # Vs/m (not provided in the paper above)
    F_R = 2.1  # guide friction force, N
    # model equations
    # Variable defintion
    x1 = SX.sym("x1")
    x2 = SX.sym("x2")
    v1 = SX.sym("v1")
    v2 = SX.sym("v2")
    I = SX.sym("I")
    # electric current
    x = vertcat(x1, v1, x2, v2, I)
    n_x = nosnoc.casadi_length(x)

    # control
    u = SX.sym("u")
    # the motor voltage

    # Dynamics
    A = np.array([
        [0, 1, 0, 0, 0],
        [-k / m1, -c_damping / m1, k / m1, c_damping / m1, K_F / m1],
        [0, 0, 0, 1, 0],
        [k / m2, c_damping / m2, -k / m2, -c_damping / m2, 0],
        [0, -K_S / L, 0, 0, -R / L],
    ])
    B = np.zeros((n_x, 1))
    B[-1, 0] = 1 / L
    C1 = np.array([0, -F_R / m1, 0, 0, 0])  # v1 >0
    C2 = -C1  # v1<0

    # switching dynamics with different friction froces
    f_1 = A @ x + B @ u + C1
    # v1>0
    f_2 = A @ x + B @ u + C2
    # v1<0

    # All modes
    F = [horzcat(f_1, f_2)]
    # Switching function
    c = [v1]
    S = [np.array([[1], [-1]])]

    # constraints
    lbu = -u_max * np.ones((1,))
    ubu = u_max * np.ones((1,))
    g_terminal = x - X_TARGET

    # Stage cost
    f_q = u**2

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, u=u)
    ocp = nosnoc.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, g_terminal=g_terminal)

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

    [model, ocp] = get_motor_with_friction_ocp_description()

    opts.terminal_time = 0.08

    solver = nosnoc.NosnocSolver(opts, model, ocp)

    results = solver.solve()
    print(f"{results['u_traj']=}")
    print(f"{results['time_steps']=}")

    return results


def example(plot=True):
    results = solve_ocp()
    if plot:
        plot_motor_with_friction(
            results["x_traj"],
            results["u_traj"],
            results["t_grid"],
            results["t_grid_u"],
        )
        plot_time_steps(results["time_steps"])


def plot_motor_with_friction(x_traj, u_traj, t_grid, t_grid_u, latexify=True):
    x_traj = np.array(x_traj)
    if latexify:
        nosnoc.latexify_plot()
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t_grid, x_traj[:, 0], label="x1")
    plt.plot(t_grid, x_traj[:, 2], label="x2")
    plt.ylabel("x")
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(t_grid, x_traj[:, 1], label="v1")
    plt.plot(t_grid, x_traj[:, 3], label="v2")
    plt.ylabel("v")
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(t_grid, x_traj[:, 4], label="I")
    plt.ylabel("I")
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.step(t_grid_u, [u_traj[0]] + u_traj, label="u")
    plt.ylabel("u")
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

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
