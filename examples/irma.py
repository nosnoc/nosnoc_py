import numpy as np
from casadi import SX, horzcat, vertcat, sum2
import matplotlib.pyplot as plt

import nosnoc

TOL = 1e-5
SWITCH_ON = 1

TSIM = 1000

# Thresholds
theta = [np.array([0.01]), np.array([0.01, 0.06, 0.08]).T, np.array([0.035]), np.array([0.04]), np.array([0.01])]
# Synthesis
kappa = np.array([[1.1e-4, 9e-4],
                  [3e-4, 0.15],
                  [6e-4, 0.018],
                  [5e-4, 0.03],
                  [7.5e-4, 0.015]])
# Degradation
gamma = np.array([0.05, 0.04, 0.05, 0.02, 0.6])

X0 = [0.011, 0.09, 0.04, 0.05, 0.015]


def get_default_options():
    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 3
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    opts.pss_mode = nosnoc.PssMode.STEP
    return opts


def get_irma_model(switch_on):
    # Variable defintion
    x = SX.sym("x", 5)

    # alphas for general inclusions
    alpha = SX.sym('alpha', 7)
    # Switching function
    c = [nosnoc.casadi_vertcat_list([x[i]-theta[i] for i in range(len(X0))])]
    # Switching multipliers
    s = horzcat(nosnoc.casadi_vertcat_list([1, 1, 1, alpha[1], 1]),
                nosnoc.casadi_vertcat_list([alpha[5], alpha[0]*(1-(1-switch_on)*(alpha[6])), alpha[2], alpha[1]*(1-alpha[4]), alpha[3]]))

    f_x = [-gamma*x + sum2(kappa*s)]
    print(f_x)

    model = nosnoc.NosnocModel(x=x, f_x=f_x, alpha=alpha, c=c, x0=X0, name='simplest_sliding')

    return model


def solve_irma(opts=None, model=None):
    if opts is None:
        opts = get_default_options()
    if model is None:
        model = get_irma_model(SWITCH_ON)

    Nsim = 500
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    return results


def plot_results(results):
    nosnoc.latexify_plot()

    plt.figure()
    for i in range(len(X0)):
        plt.subplot(5, 1, i+1)
        plt.plot(results["t_grid"], results["X_sim"][:, i])
        plt.hlines(theta[i], xmin=0, xmax=1000, linestyles='dotted')
        plt.xlim(0, 1000)
        plt.ylim(0, 1.1*max(results["X_sim"][:, i]))
        plt.ylabel(f'$x_{i+1}$')
        plt.xlabel('$t$')
    plt.show()


# EXAMPLE
def example():
    opts = get_default_options()
    opts.print_level = 1
    results = []
    model = get_irma_model(SWITCH_ON)
    results = solve_irma(opts=opts, model=model)

    plot_results(results)


if __name__ == "__main__":
    example()
