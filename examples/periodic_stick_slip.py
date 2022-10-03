import nosnoc
from casadi import SX, horzcat, vertcat
import numpy as np
import matplotlib.pyplot as plt


def get_periodic_slip_stick_model_codim1():

    # Initial value
    x0 = np.array([0.04, -0.01])

    # Variables
    x = SX.sym("x", 2)

    c = [x[1] - 0.2]

    S = [np.array([[-1], [1]])]

    f_11 = vertcat(x[1], -x[0] + 1 / (1.2 - x[1]))
    f_12 = vertcat(x[1], -x[0] - 1 / (0.8 + x[1]))

    F = [horzcat(f_11, f_12)]
    return nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)


def get_periodic_slip_stick_model_codim2():

    # Initial value
    x0 = np.array([0.04, -0.01, -0.02])

    # Variables
    x = SX.sym("x", 3)

    c = [vertcat(x[1] - 0.2, x[2] - 0.4)]

    S = [SX(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]))]

    f_11 = vertcat((x[1] + x[2]) / 2, -x[0] + 1 / (1.2 - x[1]), -x[0] + 1 / (1.4 - x[2]))
    f_12 = vertcat((x[1] + x[2]) / 2, -x[0] + 1 / (1.2 - x[1]), -x[0] + 1 / (0.6 + x[2]))
    f_13 = vertcat((x[1] + x[2]) / 2, -x[0] - 1 / (0.8 + x[1]), -x[0] + 1 / (1.4 - x[2]))
    f_14 = vertcat((x[1] + x[2]) / 2 + x[0] * (x[1] + 0.8) * (x[2] + 0.6), -x[0] - 1 / (0.8 + x[1]),
                   -x[0] - 1 / (0.6 + x[2]))
    F = [horzcat(f_11, f_12, f_13, f_14)]
    return nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)


def main_codim1():
    settings = nosnoc.NosnocSettings()

    settings.use_fesd = True
    settings.pss_mode = nosnoc.PssMode.STEWART
    settings.irk_scheme = nosnoc.IRKSchemes.RADAU_IIA
    settings.N_finite_elements = 2
    settings.n_s = 2
    settings.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
    settings.cross_comp_mode = nosnoc.CrossComplementarityMode.SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA
    settings.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    settings.comp_tol = 1e-6
    settings.equidistant_control_grid = False
    settings.print_level = 1

    Tsim = 40
    Nsim = 100
    Tstep = Tsim / Nsim
    settings.terminal_time = Tstep

    model = get_periodic_slip_stick_model_codim1()

    solver = nosnoc.NosnocSolver(settings, model)

    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()

    plot_system_codim1(results["X_sim"], results["t_grid"])
    nosnoc.plot_timings(results["cpu_nlp"])


def main_codim2():
    settings = nosnoc.NosnocSettings()

    settings.use_fesd = True
    settings.pss_mode = nosnoc.PssMode.STEWART
    settings.irk_scheme = nosnoc.IRKSchemes.RADAU_IIA
    settings.N_finite_elements = 3
    settings.n_s = 4
    settings.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
    settings.cross_comp_mode = nosnoc.CrossComplementarityMode.SUM_THETAS_COMPLEMENT_WITH_EVERY_LAMBDA
    settings.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    settings.comp_tol = 1e-9
    settings.equidistant_control_grid = False
    settings.print_level = 1

    Tsim = 20
    Nsim = 100
    Tstep = Tsim / Nsim
    settings.terminal_time = Tstep

    model = get_periodic_slip_stick_model_codim2()

    solver = nosnoc.NosnocSolver(settings, model)

    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()

    plot_system_codim2(results["X_sim"], results["t_grid"])
    nosnoc.plot_timings(results["cpu_nlp"])


def plot_system_codim1(X_sim, t_grid):
    x1 = [x[0] for x in X_sim]
    x2 = [x[1] for x in X_sim]
    # state trajectory plot
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, x1)
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(t_grid, x2)
    plt.xlabel("$t$")
    plt.ylabel("$x_2(t)$")
    plt.grid()
    # TODO figure for theta/alpha

    plt.figure()
    plt.plot(x1, x2)
    plt.xlabel("$x_1(t)$")
    plt.ylabel("$x_2(t)$")
    plt.grid()
    plt.show()


def plot_system_codim2(X_sim, t_grid):
    x1 = [x[0] for x in X_sim]
    x2 = [x[1] for x in X_sim]
    x3 = [x[2] for x in X_sim]
    # state trajectory plot
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t_grid, x1)
    plt.xlabel("$t$")
    plt.ylabel("$x_1(t)$")
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.plot(t_grid, x2)
    plt.xlabel("$t$")
    plt.ylabel("$x_2(t)$")
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.plot(t_grid, x3)
    plt.xlabel("$t$")
    plt.ylabel("$x_3(t)$")
    plt.grid()
    # TODO figure for theta/alpha

    plt.figure()
    plt.plot(x1, x3)
    plt.xlabel("$x_1(t)$")
    plt.ylabel("$x_3(t)$")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    #main_codim1()
    main_codim2()
