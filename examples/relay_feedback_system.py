import nosnoc
from casadi import SX, horzcat
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

OMEGA = 25
XI = 0.05
SIGMA = 1
NX = 3

## Info
# Simulation example from

# Piiroinen, Petri T., and Yuri A. Kuznetsov. "An event-driven method to simulate Filippov systems with
# accurate computing of sliding motions." ACM Transactions on Mathematical Software (TOMS) 34.3 (2008): 1-24.
# Equation (32)

# see also:
# M. di Bernardo, K. H. Johansson, and F. Vasca. Self-oscillations and sliding in
# relay feedback systems: Symmetry and bifurcations. International Journal of
# Bifurcations and Chaos, 11(4):1121-1140, 2001


def get_relay_feedback_system_model():

    # Initial value
    x0 = np.array([0, -0.001, -0.02])

    # Variables
    x = SX.sym("x", 3)

    A = np.array([[-(2 * XI * OMEGA + 1), 1, 0], [-(2 * XI * OMEGA + OMEGA**2), 0, 1],
                  [-OMEGA**2, 0, 0]])

    b = np.array([[1], [-2 * SIGMA], [1]])

    c = [x[0]]

    S = [np.array([[-1], [1]])]

    f_11 = A @ x + b
    f_12 = A @ x - b

    F = [horzcat(f_11, f_12)]
    return nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)


def main():
    opts = nosnoc.NosnocOpts()

    opts.use_fesd = True
    opts.pss_mode = nosnoc.PssMode.STEWART
    opts.irk_scheme = nosnoc.IRKSchemes.RADAU_IIA
    opts.N_finite_elements = 2
    opts.n_s = 2
    opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
    opts.cross_comp_mode = nosnoc.CrossComplementarityMode.SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    opts.comp_tol = 1e-6
    opts.print_level = 1
    opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.SUPERLINEAR
    opts.homotopy_update_exponent = 1.4

    Tsim = 10
    Nsim = 200

    # Tsim = 1
    # Nsim = 20
    Tstep = Tsim / Nsim
    opts.terminal_time = Tstep

    model = get_relay_feedback_system_model()

    solver = nosnoc.NosnocSolver(opts, model)
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()

    X_sim = results["X_sim"]
    t_grid = results["t_grid"]
    plot_system_trajectory(X_sim, t_grid=t_grid)
    # plot_system_3d(results)

    filename = ""
    filename = f"relay_timings_{datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S.%f')}.pdf"
    nosnoc.plot_timings(results["cpu_nlp"], figure_filename=filename)


def plot_system_3d(results):
    nosnoc.latexify_plot()

    X_sim = results["X_sim"]
    x1 = [x[0] for x in X_sim]
    x2 = [x[1] for x in X_sim]
    x3 = [x[2] for x in X_sim]
    # plot 3d curve
    plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.grid()
    plt.show()


def plot_system_trajectory(X_sim, t_grid):
    nosnoc.latexify_plot()

    # state trajectory plot
    plt.figure()
    for i in range(NX):
        plt.subplot(1, NX, i + 1)
        plt.plot(t_grid, [x[i] for x in X_sim])
        plt.grid()
        plt.xlabel("$t$")
        plt.ylabel(f"$x_{i+1}(t)$")
    plt.show()


def plot_algebraic_variables(results):
    nosnoc.latexify_plot()

    # algebraic variables
    plt.figure()
    plt.subplot(2, 1, 1)
    lambdas = [results["lambda_sim"][0][0]] + \
              [results["lambda_sim"][i][0] for i in range(len(results["lambda_sim"]))]
    thetas = [results["theta_sim"][0][0]] + \
             [results["theta_sim"][i][0] for i in range(len(results["theta_sim"]))]
    n_lam = len(lambdas[0])
    for i in range(n_lam):
        plt.plot(results["t_grid"], [x[i] for x in lambdas], label=f'$\lambda_{i+1}$')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(n_lam):
        plt.plot(results["t_grid"], [x[i] for x in thetas], label=r'$\theta_' + f'{i+1}$')
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
