import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

import nosnoc

TOL = 1e-9
TSIM = 2.0

# Initial Value
X0 = np.array([-np.sqrt(2)])


def get_simplest_model_sliding():
    # Variable defintion
    x1 = SX.sym("x1")
    x = x1
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x1]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    f_11 = 1
    f_12 = -3
    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, name='simplest_sliding')

    return model


def get_simplest_model_switch():
    # Variable defintion
    x = SX.sym("x")
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    # if x < 0
    f_11 = 1
    # if x > 0
    f_12 = 3
    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, name='simplest_switch')

    return model


def get_spontaneous_switch_model():
    # Variable defintion
    x = SX.sym("x")
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    # if x < 0
    f_11 = -1
    # if x > 0
    f_12 = 1
    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=0 * X0, name='sponteneous')

    return model


def main(variant='switch'):
    if variant not in ['switch', 'slide', 'spontaneous']:
        raise Exception(f'switch case {variant} not implemented')
    # model
    if variant == 'switch':
        model = get_simplest_model_switch()
    elif variant == 'slide':
        model = get_simplest_model_sliding()
    elif variant == 'spontaneous':
        model = get_spontaneous_switch_model()

    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 3
    opts.n_s = 2
    Nsim = 3
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL
    solver = nosnoc.NosnocSolver(opts, model)

    if variant == 'spontaneous':
        # initialize solver to leave the switching surface
        solver.set('x', np.ones((opts.N_finite_elements, 1)))
        theta_0 = np.ones((opts.N_finite_elements, 2))
        theta_0[:, 1] = 0.0
        solver.set('theta', theta_0)

    solver.problem.print()
    # loop
    looper = nosnoc.NosnocSimLooper(solver, x0=model.x0, Nsim=Nsim)
    looper.run()
    results = looper.get_results()
    # solver.print_problem()
    plot_results(results)

    return


def plot_results(results):
    nosnoc.latexify_plot()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(results["t_grid"], results["X_sim"], label='x', marker='o', markersize=3)
    plt.legend()
    plt.grid()
    # algebraic variables
    thetas = nosnoc.flatten_layer(results['theta_sim'], 0)
    thetas = [thetas[0]] + thetas

    lambdas = nosnoc.flatten_layer(results['lambda_sim'], 0)
    lambdas = [lambdas[0]] + lambdas
    n_lam = len(lambdas[0])

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


if __name__ == "__main__":
    main(variant='spontaneous')
