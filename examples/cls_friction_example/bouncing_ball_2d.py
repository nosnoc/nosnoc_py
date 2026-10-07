
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 10.0
MU = 0.3
X0 = np.array([0.0, 1.0, 4.0, 0.0])


T_SIM = 3.0
N_SIM = 500
N_FE = 2


def get_bouncing_ball_model(x0=X0):
    
    q = ca.SX.sym("q", 2)
    v = ca.SX.sym("v", 2)
    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=x0,
        M=np.eye(2),
        f_v=ca.vertcat(0.0, -GRAVITY),
        f_c=q[1],                                   
        J_tangent=ca.SX(np.array([[1.0], [0.0]])),  
        D_tangent=ca.SX(np.array([[1.0, -1.0],      
                                  [0.0, 0.0]])),    
        e=0.0,
        mu=MU,
        name="bouncing_ball_2d",
    )


def get_default_options(**kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": N_FE,
        "n_s": 1,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": False,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_FE,
        "no_initial_impacts": False,
        "friction_model": nosnoc.FrictionModel.POLYHEDRAL,
        # "friction_model": nosnoc.FrictionModel.CONIC,
        "fixed_eps_cls": True,
        "gamma_h": 1.0,
        "T":T_SIM/N_SIM,  # overwritten by the integrator with T_sim/N_sim anyway
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_integrator_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.complementarity_tol = 1e-6
    solver_opts.print_level = 3
    solver_opts.N_homotopy = 10
    solver_opts.sigma_0 = 1.0
    solver_opts.homotopy_steering_strategy = nosnoc.mpccsol.plugins.reg_homotopy.HomotopySteeringStrategy.ELL_1
    solver_opts.decreasing_s_elastic_upper_bound = True
    default_args = {
        "T_sim": T_SIM,
        "N_sim": N_SIM,
        "solver_opts": solver_opts,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def solve_bouncing_ball(x0=X0, opts=None, integrator_opts=None):
    model = get_bouncing_ball_model(x0=x0)
    if opts is None:
        opts = get_default_options()
    if integrator_opts is None:
        integrator_opts = get_default_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, _, _ = integrator.simulate(x0)
    h = opts.h_k[0] / opts.N_finite_elements[0]
    print(f"finite element length h = {h:.5f} s")
    return t_grid, x_res, integrator


def plot_results(t_grid, x_res):
    nosnoc.latexify_plot()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_res[:, 0], x_res[:, 1])
    plt.axis("equal")
    plt.grid()
    plt.xlabel("$q_x$")
    plt.ylabel("$q_y$")

    plt.subplot(1, 2, 2)
    plt.plot(t_grid, x_res[:, 3], label="$v_y$")
    plt.plot(t_grid, x_res[:, 2], label="$v_x$")
    plt.grid()
    plt.xlabel("$t$")
    plt.ylabel("$v$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def example(plot=True):
    t_grid, x_res, integrator = solve_bouncing_ball()
 
    if plot:
        plot_results(t_grid, x_res)
    return t_grid, x_res, integrator


if __name__ == "__main__":
    example()
