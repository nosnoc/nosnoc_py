import nosnoc

from casadi import SX, vertcat, horzcat, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

## Info
# This is an example from
# Stewart, D.E., 1996. A numerical method for friction problems with multiple contacts. The ANZIAM Journal, 37(3), pp.288-308.
# It considers 3 independent switching functions


def get_blocks_with_friction_model():

    ## Initial value
    x0 = np.array([-1, 1, -1, -1, 1, 1, 0])
    # u0 = 0 # guess for control variables
    ## Numer of ODE layers
    # n_simplex = 3;# number of Cartesian products in the model ("independet switches"), we call this layer
    # # number of modes in every simplex
    # m_1 = 2
    # m_2 = 2
    # m_3 = 2
    # m_vec = [m_1 m_2 m_3];

    ## Variable defintion
    # differential states
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    q3 = SX.sym('q3')
    v1 = SX.sym('v1')
    v2 = SX.sym('v2')
    v3 = SX.sym('v3')
    t = SX.sym('t')

    q = vertcat(q1, q2, q3)
    v = vertcat(v1, v2, v3)
    x = vertcat(q, v, t)

    ## Control
    # u = MX.sym('u')
    # n_u = 1;  # number of parameters,  we model it as control variables and merge them with simple equality constraints
    #
    # # Guess and Bounds
    # u0 = 0
    # lbu  = -20*0
    # ubu  = 20*0;

    ## Switching Functions
    # every constraint function corresponds to a simplex (note that the c_i might be vector valued)
    c1 = v1
    c2 = v2
    c3 = v3
    # sign matrix for the modes
    S1 = np.array([[1], [-1]])
    S2 = np.array([[1], [-1]])
    S3 = np.array([[1], [-1]])
    # discrimnant functions
    S = [S1, S2, S3]
    c = [c1, c2, c3]

    ## Modes of the ODEs layers (for all  i = 1,...,n_simplex)
    # part independet of the nonsmoothness
    F_external = 0
    # external force, e.g., control
    F_input = 10
    # variable force exicting
    f_base = vertcat(v1, v2, v3, (-q1) + (q2 - q1) - v1, (q1 - q2) + (q3 - q2) - v2,
                     (q2 - q3) - v3 + F_external + F_input * (1 * 0 + 1 * cos(np.pi * t)), 1)

    # for c1
    f_11 = f_base + vertcat(0, 0, 0, -0.3, 0, 0, 0)
    f_12 = f_base + vertcat(0, 0, 0, +0.3, 0, 0, 0)
    # for c2
    f_21 = vertcat(0, 0, 0, 0, -0.3, 0, 0)
    f_22 = vertcat(0, 0, 0, 0, 0.3, 0, 0)
    # for c3
    f_31 = vertcat(0, 0, 0, 0, 0, -0.3, 0)
    f_32 = vertcat(0, 0, 0, 0, 0, 0.3, 0)
    # in matrix form
    F1 = horzcat(f_11, f_12)
    F2 = horzcat(f_21, f_22)
    F3 = horzcat(f_31, f_32)

    F = [F1, F2, F3]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0)

    return model


def main():

    # Simulation setings
    settings = nosnoc.NosnocSettings()
    settings.n_s = 2
    settings.comp_tol = 1e-6
    settings.homotopy_update_slope = .1

    settings.N_finite_elements = 3
    Tsim = 5
    Nsim = 120
    T_step = Tsim / Nsim

    settings.terminal_time = T_step
    settings.pss_mode = nosnoc.PssMode.STEWART
    settings.irk_representation = nosnoc.IrkRepresentation.DIFFERENTIAL

    # model
    model = get_blocks_with_friction_model()

    # solver
    solver = nosnoc.NosnocSolver(settings, model)

    n_exec = 1
    for i in range(n_exec):
        # simulation loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
        looper.run()
        results = looper.get_results()
        if i == 0:
            timings = results["cpu_nlp"]
        else:
            timings = np.minimum(timings, results["cpu_nlp"])

    # evaluation
    mean_timing = np.mean(np.sum(timings, axis=1))
    print(f"mean timing solver call {mean_timing:.5f} s")

    # plots
    nosnoc.plot_timings(timings, title=settings.irk_representation)
    plot_blocks(results["X_sim"], results["t_grid"])
    import pdb
    pdb.set_trace()


def plot_blocks(X_sim, t_grid, latexify=True):

    # latexify plot
    if latexify:
        params = {
            # "backend": "TkAgg",
            "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.usetex": True,
            "font.family": "serif",
        }

        matplotlib.rcParams.update(params)

    plt.figure()
    plt.plot(t_grid, X_sim)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
