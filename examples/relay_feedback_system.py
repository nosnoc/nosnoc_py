import nosnoc
from casadi import SX, horzcat
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

OMEGA = 25
XI = 0.05
SIGMA = 1


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
    opts.equidistant_control_grid = False
    opts.print_level = 1

    Tsim = 10
    Nsim = 200
    Tstep = Tsim / Nsim
    opts.terminal_time = Tstep

    model = get_relay_feedback_system_model()

    solver = nosnoc.NosnocSolver(opts, model)

    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()

    plot_system(results["X_sim"], results["t_grid"])
    nosnoc.plot_timings(results["cpu_nlp"])


def plot_system(X_sim, t_grid):
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

    # state trajectory plot
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, x1)
    plt.plot(t_grid, x2)
    plt.plot(t_grid, x3)
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.grid()
    plt.legend(["$x_1(t)$", "$x_2(t)$", "$x_3(t)$"])
    # TODO figure for theta/alpha

    plt.show()


if __name__ == "__main__":
    main()
