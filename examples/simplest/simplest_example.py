import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

import nosnoc

TOL = 1e-9

# Analytic solution
EXACT_SWITCH_TIME = 1 / 3
TSIM = np.pi / 4
NSIM = 10

# Initial Value
X0 = np.array([-1.0])


def get_default_options(**kwargs):
    N_fe = 2
    default_args = {
        "N_stages":1,
        "N_finite_elements":N_fe,
        "T":1.0,
        "use_fesd":True,
        "cross_comp_mode":nosnoc.CrossComplementarityMode.FE_FE,
        }
    merged = dict(list(default_args.items())+ list(kwargs.items()))
    # switch
    opts = nosnoc.Options(
        **merged
    )
    return opts

def get_default_integrator_options(**kwargs):
    default_args = {
        "T_sim":TSIM,
        "N_sim":NSIM,
        }
    merged = dict(list(default_args.items())+ list(kwargs.items()))
    # switch
    opts = nosnoc.FESDIntegratorOptions(
        **merged
    )
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

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=x0, name='simplest_sliding')

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

    model = nosnoc.model.Pss(x=x, F=F, S=S, c=c, x0=X0, name='simplest_switch')

    return model


def solve_simplest_example(opts=None, model=None, integrator_opts=None, x0=X0, Nsim=1, Tsim=TSIM):
    if opts is None:
        opts = get_default_options()
        opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
        opts.pss_mode = nosnoc.DcsMode.STEWART
    if model is None:
        model = get_simplest_model_sliding()
    if integrator_opts is None:
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        integrator_opts = nosnoc.FESDIntegratorOptions(solver_opts=solver_opts, T_sim=Tsim, N_sim=Nsim, print_level=0)
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, t_grid_full, x_res_full = integrator.simulate(x0)
    # plot_results(results)
    return t_grid, x_res, t_grid_full, x_res_full, integrator


def plot_results(integrator):
    nosnoc.latexify_plot()

    t_grid = integrator.get_time_grid()
    import pdb; pdb.set_trace()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_grid, integrator.get("x"), label='x', marker='o')
    plt.legend()
    plt.grid()
    # algebraic variables
    thetas = integrator.get("theta")

    lambdas = integrator.get("lam")
    n_lam = integrator.plugin.dcs.dims.n_lambda

    plt.subplot(3, 1, 2)
    n_lam = len(lambdas[0])
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

    t_grid, x_res, t_grid_full, x_res_full, integrator = solve_simplest_example(opts=opts, model=model)

    plot_results(integrator)


if __name__ == "__main__":
    example()
