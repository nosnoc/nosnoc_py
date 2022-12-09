import nosnoc
from casadi import SX, vertcat, horzcat
import numpy as np
import matplotlib.pyplot as plt

OMEGA = 2 * np.pi
A1 = np.array([[1, OMEGA], [-OMEGA, 1]])
A2 = np.array([[1, -OMEGA], [OMEGA, 1]])
R_OSC = 1


def get_oscilator_model():

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

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)
    return model


def main():
    opts = nosnoc.NosnocOpts()
    # opts.irk_representation = "differential"
    opts.use_fesd = True
    comp_tol = 1e-6
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1  # decrease rate
    opts.N_finite_elements = 2
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED

    model = get_oscilator_model()

    Tsim = np.pi / 2
    Nsim = 29
    Tstep = Tsim / Nsim

    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()

    plot_oscilator(results["X_sim"], results["t_grid"])
    nosnoc.plot_timings(results["cpu_nlp"])
    import pdb

    pdb.set_trace()


def plot_oscilator(X_sim, t_grid, latexify=True):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, X_sim)
    plt.ylabel("$x$")
    plt.xlabel("$t$")
    plt.grid()

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
    main()
