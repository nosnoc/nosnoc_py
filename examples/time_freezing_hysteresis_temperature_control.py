"""
Time freezing example with an hysteresis curve.

In this example the temperature is controlled using a radiator.
The desired temperature is between 17 & 21 degrees and with an optimum of
19 degrees.
"""

import nosnoc
from casadi import SX, vertcat, inf, norm_2, horzcat
from math import ceil, log
import numpy as np
import matplotlib.pyplot as plt

# jump points in x in the hysteresis function
y1 = 17
y2 = 21


def create_options():
    """Create nosnoc options."""
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
    # initial smoothing parameter
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
    """
    Create temperature control model.

    :param u: input of the radiator, if this is given a simulation
        model is generated.
    """
    global y1, y2
    # Discretization parameters
    # N_stages = 2
    # N_finite_elements = 1
    # T = 0.1  # (here determined latter depeding on omega)
    # h = T/N_stages

    # inital value
    t0 = 0
    w0 = 0
    y0 = 20
    cost0 = 0
    lambda_cool_down = -0.2  # cool down time constant of lin dynamics

    # Points:
    z1 = np.array([1 / 4, -1 / 4])
    z2 = np.array([1 / 4, 1 / 4])
    z3 = np.array([3 / 4, 3 / 4])
    z4 = np.array([3 / 4, 5 / 4])

    # Define model dimensions, equations, constraint functions, regions an so on.
    # number of Cartesian products in the model ("independent switches"), we call this layer
    # Variable defintion
    y = SX.sym('y')
    w = SX.sym('w')  # Auxillary variable
    t = SX.sym('t')  # Time variable
    cost = SX.sym('cost')

    x = vertcat(y, w, t, cost)
    n_x = x.shape[0]

    # Inital Value
    X0 = np.array([y0, w0, t0, cost0]).T
    # Range
    lbx = -inf * np.ones((n_x,))
    ubx = inf * np.ones((n_x,))

    # linear transformation for rescaling of the switching function.
    psi = (y - y1) / (y2 - y1)
    z = vertcat(psi, w)

    # control
    if not u:
        u = SX.sym('u')
        s = SX.sym('s')  # Length of time
        u_comb = vertcat(u, s)
    else:
        u_comb = None
        s = 1

    lbu = np.array([1, 0.5])
    ubu = np.array([100, 20])

    # discriminant functions via voronoi
    g_11 = norm_2(z - z1)**2
    g_12 = norm_2(z - z2)**2
    g_13 = norm_2(z - z3)**2
    g_14 = norm_2(z - z4)**2

    g_ind = [vertcat(g_11, g_12, g_13, g_14)]

    # System dynamics:
    # Heating:
    y_des = 19
    f_cost = (y - y_des)**2
    f_A = vertcat(lambda_cool_down * y + u, 0, 1, f_cost)
    f_B = vertcat(lambda_cool_down * y, 0, 1, f_cost)

    a_push = 5
    f_push_down = vertcat(0, -a_push * (psi - 1)**2 / (1 + (psi - 1)**2), 0, 0)
    f_push_up = vertcat(0, a_push * (psi)**2 / (1 + (psi)**2), 0, 0)

    f_11 = s * (2 * f_A - f_push_down)
    f_12 = s * (f_push_down)
    f_13 = s * (f_push_up)
    f_14 = s * (2 * f_B - f_push_up)
    F = [horzcat(f_11, f_12, f_13, f_14)]

    # Desired temperature is 19 degrees
    f_q = 0
    f_terminal = cost

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
    plt.plot(t_grid, [y1 for _ in t_grid], 'k--')
    plt.plot(t_grid, [y2 for _ in t_grid], 'k--')
    plt.plot(t_grid, temperature, 'k', label="Temperature",)
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
    plt.figure()
    plt.plot([-2, 1], [0, 0], 'k')
    plt.plot([0, 2], [1, 1], 'k')
    plt.plot([-1, 0, 1, 2], [1.5, 1, 0, -.5], 'k')
    psi = [(x[0] - y1) / (y2 - y1) for x in X]
    im = plt.scatter(psi, aux, c=t_grid, cmap=plt.hot())
    im.set_label('Time')
    plt.colorbar(im)
    plt.xlabel("$\\psi(x)$")
    plt.ylabel("$w$")
    plt.show()


def simulation(u=20, Tsim=3, Nsim=30, with_plot=True):
    """Simulate the temperature control system with a fixed input."""
    opts = create_options()
    model = create_temp_control_model_voronoi(u=u)
    Tstep = Tsim / Nsim
    opts.N_finite_elements = 2
    opts.N_stages = 1
    opts.terminal_time = Tstep
    opts.sigma_N = 1e-2

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    if with_plot:
        plot(model, results["X_sim"], results["t_grid"])
    return results["X_sim"], results["t_grid"]


def control(with_plot=True):
    """Control the system."""
    stages = 5
    t_end = 5
    X_est, t_grid = simulation(u=1, Tsim=t_end, Nsim=stages, with_plot=False)
    X_est = np.stack(X_est[1:])

    opts = create_options()
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, X0 = create_temp_control_model_voronoi()
    opts.N_finite_elements = 3
    opts.n_s = 3
    opts.N_stages = 10
    opts.terminal_time = t_end
    opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
    opts.time_freezing = True
    opts.time_freezing_tolerance = 0.1
    opts.sigma_N = 1e-2

    ocp = nosnoc.NosnocOcp(
        lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_terminal,
        lbx=lbx, ubx=ubx
    )
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    solver.set("x", X_est)
    results = solver.solve()
    print("Dominant modes:")
    print([np.argmax(i) for i in results["theta_list"]])
    if with_plot:
        plot(model, results["x_list"], results["t_grid"][1:], results["u_list"],
             results["t_grid_u"][:-1])

    return model, opts, solver, results


if __name__ == "__main__":
    control()
