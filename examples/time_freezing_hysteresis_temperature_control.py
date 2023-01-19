import nosnoc
from casadi import SX, vertcat, inf, norm_2, horzcat
from math import ceil, log
import numpy as np
import matplotlib.pyplot as plt


def create_options():
    opts = nosnoc.NosnocOpts()
    opts.print_level = 2
    # Degree of interpolating polynomial
    opts.n_s = 3
    # === MPCC settings ===
    # upper bound for elastic variables
    opts.s_elastic_max = 1e1
    # in penalty methods  1: J = J+(1/p)*J_comp (direct)  , 0 : J = p*J+J_comp (inverse)
    opts.objective_scaling_direct = 0
    # === Penalty/Relaxation paraemetr ===
    # starting smouothing parameter
    opts.sigma_0 = 1e1
    # end smoothing parameter
    opts.sigma_N = 1e-3  # 1e-10
    # decrease rate
    opts.homotopy_update_slope = 0.1
    # number of steps
    opts.N_homotopy = ceil(abs(
        log(opts.sigma_N / opts.sigma_0) / log(opts.homotopy_update_slope))) + 1
    opts.comp_tol = 1e-14

    # IPOPT Settings
    opts.opts_casadi_nlp['ipopt']['max_iter'] = 500

    # New setting: time freezing settings
    opts.initial_theta = 0.5
    opts.time_freezing = False
    opts.pss_mode = nosnoc.PssMode.STEWART
    return opts


def create_temp_control_model_voronoi(u=None):
    # Discretization parameters
    # N_stages = 2
    # N_finite_elements = 1
    # T = 0.1  # (here determined latter depeding on omega)
    # h = T/N_stages

    # inital value
    t0 = 0
    w0 = 1
    y0 = 22

    lambda_cool_down = -0.2  # cool down time constant of lin dynamics

    # jump points in x in the hysteresis function
    y1 = 18
    y2 = 20

    z1 = np.array([1 / 4, -1 / 4])
    z2 = np.array([1 / 4, 1 / 4])
    z3 = np.array([3 / 4, 3 / 4])
    z4 = np.array([3 / 4, 5 / 4])
    # Z = [1/4 1/4 3/4 3/4;...
    #      -1/4 1/4 3/4 5/4]
    # Z = np.concatenate([z1, z2, z3, z4])

    # Inital Value
    X0 = np.array([y0, w0, t0]).T

    # Define model dimensions, equations, constraint functions, regions an so on.
    # number of Cartesian products in the model ("independet switches"), we call this layer
    # Variable defintion
    y = SX.sym('y')
    w = SX.sym('w')  # Auxillary variable
    t = SX.sym('t')  # Time variable

    x = vertcat(y, w, t)
    n_x = x.shape[0]
    lbx = -inf * np.ones((n_x,))
    ubx = inf * np.ones((n_x,))

    # linear transformation for rescaling of the switching function.
    psi = (y - y1) / (y2 - y1)
    # z = vertcat(psi, w)

    # discriminant functions via voronoi
    g_11 = (psi - z1[0])**2 + (w - z1[1])**2
    g_12 = (psi - z2[0])**2 + (w - z2[1])**2
    g_13 = (psi - z3[0])**2 + (w - z3[1])**2
    g_14 = (psi - z4[0])**2 + (w - z4[1])**2

    g_ind = [vertcat(g_11, g_12, g_13, g_14)]

    # control
    if not u:
        u = SX.sym('u')
        s = SX.sym('s')  # Length of time
        u_comb = vertcat(u, s)
    else:
        u_comb = None
        s = 1

    lbu = np.array([0, 1])
    ubu = np.array([1, 1])

    # System dynamics:
    # Heating:
    f_A = s * vertcat(lambda_cool_down * y + u, 0, 1)
    f_B = s * vertcat(lambda_cool_down * y, 0, 1)

    a_push = 5
    f_push_down = vertcat(0, -a_push * (psi - 1)**2 / (1 + (psi - 1)**2), 0)
    f_push_up = vertcat(0, a_push * (psi)**2 / (1 + (psi)**2), 0)

    f_11 = 2 * f_A - f_push_down
    f_12 = f_push_down
    f_13 = f_push_up
    f_14 = 2 * f_B - f_push_up
    F = [horzcat(f_11, f_12, f_13, f_14)]

    # Desired temperature is 19 degrees
    y_des = 19
    f_q = (y-y_des)**2
    f_terminal = 0  # (y - y_des)**2

    if u_comb is not None:
        model = nosnoc.NosnocModel(x=x,
                                   F=F,
                                   g_Stewart=g_ind,
                                   x0=X0,
                                   u=u_comb,
                                   t_var=t,
                                   name='simplest_sliding')
        return model, lbx, ubx, lbu, ubu, f_q, f_terminal, X0
    else:
        model = nosnoc.NosnocModel(
            x=x, F=F, g_Stewart=g_ind, x0=X0, name='simplest_sliding')
        return model


def plot(model, X, t_grid, U=None, t_grid_u=None):
    """Plot the results."""
    temperature = [x[0] for x in X]
    aux = [x[1] for x in X]
    time = [x[2] for x in X]

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t_grid, temperature, label="Temperature")
    plt.plot(t_grid, aux, label="Auxillary variable")
    plt.plot(t_grid, time, label="Time")
    plt.ylabel("$x$")
    plt.xlabel("$t$")
    plt.legend()
    plt.grid()

    if U is not None:
        plt.subplot(2, 2, 2)
        plt.plot(t_grid_u, [u[0] for u in U], label="Control")
        plt.plot(t_grid_u, [u[1] for u in U], label="Time scaling")
        plt.ylabel("$u$")
        plt.xlabel("$t$")
        plt.legend()
        plt.grid()

    plt.subplot(2, 2, 3)
    plt.ylabel("Temperature")
    plt.xlabel("Real time")
    plt.plot(time, temperature, label="Temperature in real time")

    plt.subplot(2, 2, 4)
    plt.ylabel("G")
    plt.xlabel("Time")
    g = horzcat(*[model.g_Stewart_fun(x, 0) for x in X]).T
    plt.plot(t_grid, g, label=[f"mode {i}" for i in range(g.shape[1])])
    plt.legend()
    plt.show()


def simulation(u=20, Tsim=3, Nsim=30, with_plot=True):
    """Simulate the temperature control system with a fixed input."""
    opts = create_options()
    model = create_temp_control_model_voronoi(u=20)
    Tstep = Tsim / Nsim
    opts.N_finite_elements = 2
    opts.N_stages = 1
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    if with_plot:
        plot(model, results["X_sim"], results["t_grid"])


def control():
    """Control the system."""
    opts = create_options()
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, X0 = create_temp_control_model_voronoi()
    opts.N_finite_elements = 3
    opts.N_stages = 10
    opts.terminal_time = opts.N_stages
    opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
    opts.time_freezing = True
    X_est = np.repeat(X0.reshape((1, -1)), opts.N_stages, axis=0)
    X_est[:, 2] = np.linspace(1, opts.N_stages+1, opts.N_stages)

    ocp = nosnoc.NosnocOcp(
        lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_terminal,
        lbx=lbx, ubx=ubx
    )
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    solver.set("x", X_est)
    results = solver.solve()
    plot(model, results["x_list"], results["t_grid"][1:], results["u_list"],
         results["t_grid_u"][:-1])


if __name__ == "__main__":
    control()
