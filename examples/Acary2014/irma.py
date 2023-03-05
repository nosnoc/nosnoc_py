import numpy as np
from casadi import SX, horzcat, sum2, vertcat
import matplotlib.pyplot as plt

import nosnoc

# Example synthetic benchmark from:
# Numerical simulation of piecewise-linear models of gene regulatory networks using complementarity systems
# V. Acary, H. De Jong, B. Brogliato

TOL = 1e-5
SWITCH_ON = 1

TSIM = 1000

# Thresholds
thresholds = [np.array([0.01]), np.array([0.01, 0.06, 0.08]).T, np.array([0.035]), np.array([0.04]), np.array([0.01])]
# Synthesis
kappa = np.array([[1.1e-4, 9e-4],
                  [3e-4, 0.15],
                  [6e-4, 0.018],
                  [5e-4, 0.03],
                  [7.5e-4, 0.015]])
# Degradation
gamma = np.array([0.05, 0.04, 0.05, 0.02, 0.6])

X0 = [0.011, 0.09, 0.04, 0.05, 0.015]
LIFTING = True


def get_default_options():
    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 3
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    opts.pss_mode = nosnoc.PssMode.STEP
    opts.print_level = 0
    opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.LINEAR

    return opts


def get_irma_model(switch_on, lifting):
    # Variable defintion
    x = SX.sym("x", 5)

    # alphas for general inclusions
    alpha = SX.sym('alpha', 7)
    # Switching function
    c = [nosnoc.casadi_vertcat_list([x[i]-thresholds[i] for i in range(len(X0))])]
    if lifting:
        if switch_on:
            beta = SX.sym('beta', 1)
            g_z = beta - alpha[1]*(1-alpha[4])

            s = horzcat(nosnoc.casadi_vertcat_list([1, 1, 1, alpha[1], 1]),
                        nosnoc.casadi_vertcat_list([alpha[5], alpha[0], alpha[2], beta, alpha[3]]))
        else:
            beta = SX.sym('beta', 2)
            g_z = beta - vertcat(alpha[0]*(1-alpha[6]), alpha[1]*(1-alpha[4]))

            s = horzcat(nosnoc.casadi_vertcat_list([1, 1, 1, alpha[1], 1]),
                        nosnoc.casadi_vertcat_list([alpha[5], beta[0], alpha[2], beta[1], alpha[3]]))
        f_x = [-gamma*x + sum2(kappa*s)]
        model = nosnoc.NosnocModel(x=x, f_x=f_x, g_z=g_z, z=beta, alpha=[alpha], c=c, x0=X0, name='irma')
    else:
        # Switching multipliers
        s = horzcat(nosnoc.casadi_vertcat_list([1, 1, 1, alpha[1], 1]),
                    nosnoc.casadi_vertcat_list([alpha[5], alpha[0]*(1-(1-switch_on)*(alpha[6])), alpha[2], alpha[1]*(1-alpha[4]), alpha[3]]))
        f_x = [-gamma*x + sum2(kappa*s)]
        model = nosnoc.NosnocModel(x=x, f_x=f_x, alpha=[alpha], c=c, x0=X0, name='irma')

    return model


def solve_irma(opts=None, model=None):
    if opts is None:
        opts = get_default_options()
    if model is None:
        model = get_irma_model(SWITCH_ON, LIFTING)

    Nsim = 500
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    return results


def plot_trajectory(results, figure_filename=''):
    nosnoc.latexify_plot()

    n_subplot = len(X0)
    fig, axs = plt.subplots(n_subplot, 1)
    for i in range(n_subplot):
        axs[i].plot(results["t_grid"], results["X_sim"][:, i])
        axs[i].hlines(thresholds[i], xmin=0, xmax=TSIM, linestyles='dotted')
        axs[i].set_xlim(0, TSIM)
        axs[i].set_ylim(0, 1.1*max(results["X_sim"][:, i]))
        axs[i].set_ylabel(f'$x_{i+1}$')
        axs[i].grid()
        if i == n_subplot - 1:
            plt.xlabel('$t$')
        else:
            axs[i].xaxis.set_ticklabels([])

    if figure_filename != '':
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    plt.show()

def plot_algebraic_traj(results, figure_filename=''):
    nosnoc.latexify_plot()
    alpha_sim = np.array([results['alpha_sim'][0][0]] + nosnoc.flatten_layer(results['alpha_sim']))
    n_subplot = len(alpha_sim[0])

    fig, axs = plt.subplots(n_subplot, 1)
    for i in range(n_subplot):
        axs[i].plot(results["t_grid"], alpha_sim[:,i])
        # axs[i].hlines(thresholds[i], xmin=0, xmax=TSIM, linestyles='dotted')
        axs[i].set_xlim(0, TSIM)
        axs[i].set_ylim(0, 1.1*max(alpha_sim[:, i]))
        axs[i].set_ylabel(r'$\alpha_' + f'{i+1}$')
        axs[i].set_xlabel('$t$')
        axs[i].grid()
        if i == n_subplot - 1:
            axs[i].set_xlabel('$t$')
        else:
            axs[i].xaxis.set_ticklabels([])

    if figure_filename != '':
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    plt.show()

# EXAMPLE
def example():
    opts = get_default_options()
    opts.print_level = 1

    model = get_irma_model(SWITCH_ON, LIFTING)
    results = solve_irma(opts=opts, model=model)

    plot_algebraic_traj(results)
    plot_trajectory(results)

    # plot_algebraic_traj(results, figure_filename='irma_algebraic_traj.pdf')
    # plot_trajectory(results, figure_filename='irma_traj.pdf')


if __name__ == "__main__":
    example()
