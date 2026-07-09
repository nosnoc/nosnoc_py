import numpy as np
from casadi import SX, horzcat, vertcat
import matplotlib.pyplot as plt

import nosnoc

# Example gene network from:
# Numerical simulation of piecewise-linear models of gene regulatory networks using complementarity systems
# V. Acary, H. De Jong, B. Brogliato

TOL = 1e-9

TSIM = 1

# Thresholds
thresholds_1 = np.array([4, 8])
thresholds_2 = np.array([4, 8])
# Synthesis
kappa = np.array([40, 40])
# Degradation
gamma = np.array([4.5, 1.5])


X0 = np.array([9, 9])
LIFTING = False

def get_default_options(**kwargs) -> nosnoc.Options:
    default_args = {
        "N_finite_elements": 4,
        "n_s": 2,
        "T":1.0,
        "use_fesd":True,
        "cross_comp_mode":nosnoc.CrossComplementarityMode.FE_FE,
        "rho_h": 1.0,
        }
    merged = dict(list(default_args.items())+ list(kwargs.items()))
    opts = nosnoc.Options(
        **merged
    )
    return opts

def get_two_gene_model(x0, lifting):
    # Variable defintion
    x = SX.sym("x", 2)

    # alphas for general inclusions
    alpha = SX.sym('alpha', 4)
    # Switching function
    c = vertcat(x[0]-thresholds_1, x[1]-thresholds_2)
    # Switching multipliers
    s = vertcat((1-alpha[1])*alpha[2], alpha[0]*(1-alpha[3]))
    if lifting:
        beta = SX.sym('beta', 2)
        g_z = beta - s
        f_x = -gamma*x + kappa*beta

        model = nosnoc.model.Heaviside(x=x, f_x=f_x, z=beta, g_z=g_z, alpha=alpha, c=c, x0=x0, name='two_gene')
    else:
        f_x = -gamma*x + kappa*s
        model = nosnoc.model.Heaviside(x=x, f_x=f_x, alpha=alpha, c=c, x0=x0, name='two_gene')

    return model


def solve_two_gene(opts=None, integrator_opts=None, model=None):
    Nsim = 30
    if opts is None:
        opts = get_default_options()
    if model is None:
        model = get_two_gene_model(X0, False)
    if integrator_opts is None:
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        solver_opts.homotopy_steering_strategy = nosnoc.mpccsol.plugins.reg_homotopy.HomotopySteeringStrategy.ELL_INF
        integrator_opts = nosnoc.FESDIntegratorOptions(solver_opts=solver_opts, T_sim=TSIM, N_sim=Nsim, print_level=3)

    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, t_grid_full, x_res_full = integrator.simulate(model.x0)
    return t_grid, x_res, t_grid_full, x_res_full, integrator

def plot_results(integrators):
    nosnoc.latexify_plot()

    plt.figure()
    for integrator in integrators:
        x_res = integrator.get("x")
        plt.plot(x_res[:, 0], x_res[:, 1])
        plt.quiver(x_res[:-1, 0],
                   x_res[:-1, 1],
                   np.diff(x_res[:, 0]),
                   np.diff(x_res[:, 1]),
                   scale=100,
                   width=0.01)
    plt.vlines(thresholds_1, ymin=-15.0, ymax=15.0, linestyles='dotted')
    plt.hlines(thresholds_2, xmin=-15.0, xmax=15.0, linestyles='dotted')
    plt.ylim(0, 13)
    plt.xlim(0, 13)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()


# EXAMPLE
def example():
    opts = get_default_options()
    opts.print_level = 0
    integrators = []
    for x1 in [3, 5, 9, 12]:
        for x2 in [3, 5, 9, 12]:
            model = get_two_gene_model(np.array([x1, x2]), LIFTING)
            t_grid, x_res, t_grid_full, x_res_full, integrator = solve_two_gene(opts=opts, model=model)
            integrators.append(integrator)

    plot_results(integrators)


if __name__ == "__main__":
    example()
