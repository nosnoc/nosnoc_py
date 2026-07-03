from copy import copy
import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

import nosnoc
import casadi as ca
from nosnoc.nosnoc_types import CrossComplementarityMode

TOL = 1e-9

# Analytic solution
EXACT_SWITCH_TIME = 1 / 3
TSIM = np.pi / 4
NSIM = 10

# Initial Value
X0 = np.array([-1.0])


def get_default_options():
    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 2
    opts.n_s = 2
    opts.print_level = 1
    return opts


def get_simplest_model_sliding(x0=X0):
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

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=x0, lbx=-5)


    return model


def get_simplest_model_switch(x0=X0):
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

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=x0, lbx=-5)

    return model


def solve_simplest_example(opts=None, model=None, x0=X0, Nsim=1, Tsim=TSIM):
    if opts is None:
        opts = get_default_options()
        opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
        opts.pss_mode = nosnoc.PssMode.STEWART
    if model is None:
        model = get_simplest_model_sliding()

    Tstep = Tsim / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)
    # loop
    looper = nosnoc.NosnocSimLooper(solver, x0, Nsim)
    looper.run()
    results = looper.get_results()
    # solver.print_problem()
    # plot_results(results)
    return results


def plot_results(solver):
    nosnoc.latexify_plot()
    t_grid = solver.get_time_grid()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_grid, solver.get("x"), label='x', marker='o')
    plt.legend()
    plt.grid()
    # algebraic variables
    thetas = solver.get("theta")

    lambdas = solver.get("lam")

    plt.subplot(3, 1, 2)
    n_lam = lambdas.shape[1]
    for i in range(n_lam):
        plt.plot(t_grid, lambdas[:,i], label=f'lambda_{i}')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    for i in range(n_lam):
        plt.plot(t_grid, thetas[:,i], label=f'theta_{i}')
    plt.grid()
    plt.vlines(t_grid, ymin=0.0, ymax=1.0, linestyles='dotted')
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
    model_sliding = get_simplest_model_sliding()
    model_switch = get_simplest_model_switch()

    N_stages = 10
    N_fe = 2

    # switch
    opts = nosnoc.Options(
        N_stages=N_stages,
        N_finite_elements=[N_fe]*N_stages,
        h_k=[1/(N_fe*N_stages)]*N_stages,
        x_box_at_stg=False,
        x_box_at_fe=False,
        use_fesd=True,
        cross_comp_mode=CrossComplementarityMode.FE_FE
    )
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver = nosnoc.OcpSolver(model_switch, opts, solver_opts)
    solver.set("x", (slice(1,None), slice(1,None), slice(1,None)), lb=-10, ub=10, init=0)
    solver.solve()
    plot_results(solver)

    # sliding
    opts = nosnoc.Options(
        N_stages=N_stages,
        N_finite_elements=[N_fe]*N_stages,
        h_k=[1/(N_fe*N_stages)]*N_stages,
        x_box_at_stg=False,
        x_box_at_fe=False,
        use_fesd=True,
        cross_comp_mode=CrossComplementarityMode.FE_FE
    )
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver = nosnoc.OcpSolver(model_sliding, opts, solver_opts)
    solver.solve()
    plot_results(solver)

    # integrator
    opts = nosnoc.Options(
        N_stages=1,
        N_finite_elements=[N_fe],
        T=TSIM/NSIM,
        N_sim=NSIM,
        h_k=[1/(N_fe)],
        x_box_at_stg=False,
        x_box_at_fe=False,
        use_fesd=True,
        cross_comp_mode=CrossComplementarityMode.FE_FE
    )
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    integrator_opts = nosnoc.FESDIntegratorOptions(solver_opts=solver_opts)
    integrator = nosnoc.Integrator(model_switch, opts, integrator_opts)
    dtp = integrator.plugin.dtp
    dtp.w.resort_vector()
    dtp.g.resort_vector()
    dtp.p.resort_vector()
    t_grid, x_res, t_grid_full, x_res_full = integrator.simulate(X0)
    import pdb; pdb.set_trace()
