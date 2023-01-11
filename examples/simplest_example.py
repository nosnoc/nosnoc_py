import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

import nosnoc

TOL = 1e-9

# Analytic solution
EXACT_SWITCH_TIME = 1 / 3
TSIM = np.pi / 4

# Initial Value
X0 = np.array([-1.0])


def get_default_options():
    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 2
    opts.n_s = 2
    return opts


def get_simplest_model_sliding():
    # Variable defintion
    x1 = SX.sym("x1")
    x = x1
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x1]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    f_11 = 3
    f_12 = -1
    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, name='simplest_sliding')

    return model


def get_simplest_model_switch():
    # Variable defintion
    x1 = SX.sym("x1")
    x = x1
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x1]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    f_11 = 3
    f_12 = 1
    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, name='simplest_switch')

    return model


def solve_simplest_example(opts=None, model=None):
    if opts is None:
        opts = get_default_options()
        opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
        opts.pss_mode = nosnoc.PssMode.STEWART
    if model is None:
        model = get_simplest_model_sliding()

    Nsim = 1
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, X0, Nsim)
    looper.run()
    results = looper.get_results()
    # solver.print_problem()
    # plot_results(results)
    return results


def plot_results(results):
    nosnoc.latexify_plot()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(results["t_grid"], results["X_sim"], label='x', marker='o')
    plt.legend()
    plt.grid()

    # algebraic variables
    lambdas = [results["lambda_sim"][0][0]] + \
              [results["lambda_sim"][i][0] for i in range(len(results["lambda_sim"]))]
    thetas = [results["theta_sim"][0][0]] + \
             [results["theta_sim"][i][0] for i in range(len(results["theta_sim"]))]
    plt.subplot(3, 1, 2)
    n_lam = len(lambdas[0])
    for i in range(n_lam):
        plt.plot(results["t_grid"], [x[i] for x in lambdas], label=f'lambda_{i}')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    for i in range(n_lam):
        plt.plot(results["t_grid"], [x[i] for x in thetas], label=f'theta_{i}')
    plt.grid()
    plt.vlines(results["t_grid"], ymin=0.0, ymax=1.0, linestyles='dotted')
    plt.legend()
    plt.show()


# EXAMPLE
def example():
    model = get_simplest_model_sliding()
    model = get_simplest_model_switch()

    opts = get_default_options()
    opts.print_level = 1

    results = solve_simplest_example(opts=opts, model=model)

    plot_results(results)


if __name__ == "__main__":
    example()
