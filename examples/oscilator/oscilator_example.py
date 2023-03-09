import nosnoc
from casadi import SX, vertcat, horzcat
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

OMEGA = 2 * np.pi
A1 = np.array([[1, OMEGA], [-OMEGA, 1]])
A2 = np.array([[1, -OMEGA], [OMEGA, 1]])
R_OSC = 1

TSIM = np.pi / 2
NSIM = 29
TSTEP = TSIM / NSIM

X_SOL = np.array([np.exp(TSIM-1) * np.cos(2*np.pi * (TSIM-1)), -np.exp(TSIM-1) * np.sin(2*np.pi*(TSIM-1))])

def get_oscilator_model(use_g_Stewart=False):

    # Initial Value
    x0 = np.array([np.exp([-1])[0], 0])

    # Variable defintion
    x1 = SX.sym("x1")
    x2 = SX.sym("x2")
    x = vertcat(x1, x2)
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x1**2 + x2**2 - R_OSC**2]
    # sign matrix for the modes
    S = [np.array([[1], [-1]])]

    f_11 = A1 @ x
    f_12 = A2 @ x
    # in matrix form
    F = [horzcat(f_11, f_12)]

    if use_g_Stewart:
        g_Stewart_list = [-S[i] @ c[i] for i in range(1)]
        model = nosnoc.NosnocModel(x=x, F=F, g_Stewart=g_Stewart_list, x0=x0)
    else:
        model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)

    return model

def get_default_options():
    opts = nosnoc.NosnocOpts()
    comp_tol = 1e-8
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1  # decrease rate
    opts.N_finite_elements = 2
    opts.n_s = 3
    opts.print_level = 1
    opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER

    return opts


def solve_oscilator(opts=None, use_g_Stewart=False, do_plot=True):
    if opts is None:
        opts = get_default_options()

    model = get_oscilator_model(use_g_Stewart)
    opts.terminal_time = TSTEP
    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, NSIM)
    looper.run()
    results = looper.get_results()

    error = np.max(np.abs(X_SOL - results["X_sim"][-1]))
    print(f"error wrt exact solution {error:.2e}")

    if do_plot:
        plot_oscilator(results["X_sim"], results["t_grid"], switch_times=results["switch_times"])
    nosnoc.plot_timings(results["cpu_nlp"])

    # store solution
    # import json
    # json_file = 'oscilator_results_ref.json'
    # with open(json_file, 'w') as f:
    #     json.dump(results['w_sim'], f, indent=4, sort_keys=True, default=make_object_json_dumpable)
    # print(f"saved results in {json_file}")
    return results

def main_least_squares():

    # load reference solution
    # import json
    # json_file = 'oscilator_results_ref.json'
    # with open(json_file, 'r') as f:
    #     w_sim_ref = json.load(f)

    opts = nosnoc.NosnocOpts()
    comp_tol = 1e-7
    opts.comp_tol = comp_tol
    opts.print_level = 2

    # opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.SUPERLINEAR
    opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
    opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER
    opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
    opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
    opts.initialization_strategy = nosnoc.InitializationStrategy.ALL_XCURRENT_W0_START
    opts.initialization_strategy = nosnoc.InitializationStrategy.RK4_SMOOTHENED
    opts.sigma_0 = 1e0
    # opts.gamma_h = np.inf
    # opts.nlp_max_iter = 0
    # opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.SUPERLINEAR
    opts.homotopy_update_slope = 0.1

    model = get_oscilator_model()

    opts.terminal_time = TSTEP

    solver = nosnoc.NosnocSolver(opts, model)
    solver.print_problem()
    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, NSIM)
    # looper = nosnoc.NosnocSimLooper(solver, model.x0, NSIM, w_init=w_sim_ref)
    looper.run()
    results = looper.get_results()
    print(f"max cost_val = {max(results['cost_vals']):.2e}")

    error = np.max(np.abs(X_SOL - results["X_sim"][-1]))
    print(f"error wrt exact solution {error:.2e}")

    plot_oscilator(results["X_sim"], results["t_grid"])
    # nosnoc.plot_timings(results["cpu_nlp"])


def main_custom_solver():

    opts = get_default_options()

    model = get_oscilator_model()

    opts.terminal_time = TSTEP
    # opts.print_level = 2
    # opts.pss_mode = nosnoc.PssMode.STEP
    # opts.init_lambda = .5

    # opts.initialization_strategy = nosnoc.InitializationStrategy.ALL_XCURRENT_WOPT_PREV
    # opts.initialization_strategy = nosnoc.InitializationStrategy.RK4_SMOOTHENED
    # opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.SUPERLINEAR
    # opts.sigma_0 = 1e-2

    solver = nosnoc.NosnocCustomSolver(opts, model)
    # solver = nosnoc.NosnocSolver(opts, model)
    # solver.print_problem()
    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, NSIM)
    # looper = nosnoc.NosnocSimLooper(solver, model.x0, NSIM, w_init=w_sim_ref)
    looper.run()
    results = looper.get_results()
    print(f"max cost_val = {max(results['cost_vals']):.2e}")

    error = np.max(np.abs(X_SOL - results["X_sim"][-1]))
    print(f"error wrt exact solution {error:.2e}")

    plot_oscilator(results["X_sim"], results["t_grid"])
    timings = results["cpu_nlp"]
    mean_cpu = np.mean(np.sum(timings, axis=1))
    print(f"Mean CPU time: {mean_cpu:.3f} s")
    nosnoc.plot_timings(timings)

def main_polishing():

    opts = get_default_options()
    opts.comp_tol = 1e-4
    opts.do_polishing_step = True

    opts.cross_comp_mode = nosnoc.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
    opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
    opts.constraint_handling = nosnoc.ConstraintHandling.LEAST_SQUARES
    opts.mpcc_mode = nosnoc.MpccMode.FISCHER_BURMEISTER
    opts.print_level = 3

    results = solve_oscilator(opts, do_plots=False)
    print(f"max cost_val = {max(results['cost_vals']):.2e}")

    nosnoc.plot_timings(results["cpu_nlp"])


def plot_oscilator(X_sim, t_grid, latexify=True, switch_times=None):
    if latexify:
        nosnoc.latexify_plot()

    # trajectory
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, X_sim)
    plt.ylabel("$x$")
    plt.xlabel("$t$")
    if switch_times is not None:
        for t in switch_times:
            plt.axvline(t, linestyle="dashed", color="k")
    plt.grid()

    # state space plot
    ax = plt.subplot(1, 2, 2)
    plt.ylabel("$x_2$")
    plt.xlabel("$x_1$")
    x1 = [x[0] for x in X_sim]
    x2 = [x[1] for x in X_sim]
    plt.plot(x1, x2)
    ax.add_patch(plt.Circle((0, 0), 1.0, color="r", fill=False))

    # vector field
    width = 2.0
    n_grid = 20
    x, y = np.meshgrid(np.linspace(-width, width, n_grid), np.linspace(-width, width, n_grid))

    indicator = np.sign(x**2 + y**2 - R_OSC**2)
    u = (A1[0, 0] * x + A1[0, 1] * y) * 0.5 * (indicator + 1) + (
        A2[0, 0] * x + A2[0, 1] * y) * 0.5 * (1 - indicator)
    v = (A1[1, 0] * x + A1[1, 1] * y) * 0.5 * (indicator + 1) + (
        A2[1, 0] * x + A2[1, 1] * y) * 0.5 * (1 - indicator)

    plt.quiver(x, y, u, v)

    plt.show()


if __name__ == "__main__":
    solve_oscilator(use_g_Stewart=False, do_plot=True)
    main_custom_solver()
    # main_least_squares()
    # main_polishing()
