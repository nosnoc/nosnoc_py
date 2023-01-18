import nosnoc
from casadi import SX, vertcat, inf, norm_2, horzcat
from math import ceil, log
import numpy as np
import matplotlib.pyplot as plt


def create_options():
    opts = nosnoc.NosnocOpts()
    # Degree of interpolating polynomial
    opts.n_s = 1
    # === MPCC settings ===
    # upper bound for elastic variables
    opts.s_elastic_max = 1e1
    # in penalty methods  1: J = J+(1/p)*J_comp (direct)  , 0 : J = p*J+J_comp (inverse)
    opts.objective_scaling_direct = 0
    # === Penalty/Relaxation paraemetr ===
    # starting smouothing parameter
    opts.sigma_0 = 1e1
    # end smoothing parameter
    opts.sigma_N = 1e-3 # 1e-10
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
    w0 = 0
    y0 = 15

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
    w = SX.sym('w')
    t = SX.sym('t')

    x = vertcat(y, w, t)
    n_x = x.size()
    lbx = -inf * np.ones(n_x)
    ubx = inf * np.ones(n_x)

    # linear transformation for rescaling of the switching function.
    psi = (y - y1) / (y2 - y1)
    z = vertcat(psi, w)

    # discriminant functions via voronoi
    g_11 = (psi - z1[0]) ** 2 + (w - z1[1]) ** 2
    g_12 = (psi - z2[0]) ** 2 + (w - z2[1]) ** 2
    g_13 = (psi - z3[0]) ** 2 + (w - z3[1]) ** 2
    g_14 = (psi - z4[0]) ** 2 + (w - z4[1]) ** 2

    # g_11 = -2* z @ z1 + z1.T @ z1;
    # g_12 = -2* z @ z2 + z2.T @ z2;
    # g_13 = -2* z @ z3 + z3.T @ z3;
    # g_14 = -2* z @ z4 + z4.T @ z4;

    g_ind = [vertcat(g_11, g_12, g_13, g_14)]

    # control
    if not u:
        u = SX.sym('u')
        s = SX.sym('s')  # Length of time
        u_comb = vertcat(u, s)
    else:
        u_comb = None
        s = 1

    umax = 1e-3

    lbu = np.array([-umax, 1])
    ubu = np.array([umax, 10])

    # System dynamics:
    # Heating:
    f_A = s * vertcat(lambda_cool_down * y + u, 0, 1)
    f_B = s * vertcat(lambda_cool_down * y, 0, 1)

    a_push = 5
    f_push_down = vertcat(
        0, -a_push * (psi - 1) ** 2 / (1 + (psi - 1) ** 2), 0
    )
    f_push_up = vertcat(
        0, a_push * (psi) ** 2 / (1 + (psi) ** 2), 0
    )

    f_11 = 2* f_A - f_push_down
    f_12 = f_push_down
    f_13 = f_push_up
    f_14 = 2* f_B - f_push_up
    F = [horzcat(f_11, f_12, f_13, f_14)]

    # objective
    f_q = (u ** 2) + y ** 2
    f_terminal = 0  # TODO: Add cost!

    if u_comb:
        model = nosnoc.NosnocModel(x=x, F=F, g_Stewart=g_ind, x0=X0, u=u_comb, name='simplest_sliding')
        return model, lbx, ubx, lbu, ubu, f_q, f_terminal
    else:
        model = nosnoc.NosnocModel(x=x, F=F, g_Stewart=g_ind, x0=X0, name='simplest_sliding')
        return model

def plot(results):
    """Plot the results."""
    X_sim = results["X_sim"]
    temperature = [x[0] for x in X_sim]
    aux = [x[1] for x in X_sim]
    time = [x[2] for x in X_sim]
    t_grid = results["t_grid"]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t_grid, temperature, label="Temperature")
    plt.plot(t_grid, aux, label="Auxillary variable")
    plt.plot(t_grid, time, label="Time")
    plt.ylabel("$x$")
    plt.xlabel("$t$")
    plt.legend()
    plt.grid()

    ax = plt.subplot(1, 3, 2)
    plt.ylabel("Temperature")
    plt.xlabel("Real time")
    plt.plot(time, temperature, label="Temperature in real time")

    ax = plt.subplot(1, 3, 3)
    plt.ylabel("G")
    plt.xlabel("Time")
    g = horzcat(*[model.g_Stewart_fun(x, 0) for x in X_sim]).T
    plt.plot(t_grid, g)
    plt.show()


opts = create_options()
model = create_temp_control_model_voronoi(u=20)
Tsim = 3
Nsim = 30
Tstep = Tsim / Nsim
opts.N_finite_elements = 2
opts.N_stages = 1
opts.terminal_time = Tstep

solver = nosnoc.NosnocSolver(opts, model)


# loop
looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
looper.run()
results = looper.get_results()
# model.g_Stewart_fun([results["X_sim"], 0])
plot(results)
# # Generate Model
# model = temp_control_model_voronoi()
# # - Simulation settings
# model.T_sim = 3
# model.N_stages = 1
# model.N_finite_elements = 2
# model.N_sim = 30
# settings.use_previous_solution_as_initial_guess = 1
# # Call FESD Integrator
# [results, stats] = integrator_fesd(model, settings)
