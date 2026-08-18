# Disc OCP

import nosnoc as ns
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def get_particle_model(opts, friction, x0):
    g = 10
    a_n = 20
    mu = 0.2
    # Symbolic variables
    q = ca.SX.sym('q', 2)
    v = ca.SX.sym('v', 2)
    x = ca.vertcat(q, v)
    t = ca.SX.sym('t', 1)

    if friction:
        # multipliers
        alpha = ca.SX.sym('alpha', 3)
        theta = ca.SX.sym('theta', 3)
        z = theta 
        z0 = np.array([0, 0, 0])

        theta_expr = ca.vertcat((alpha[0]+alpha[1])-(alpha[0]*alpha[1]),
                                (1-alpha[0])*(1-alpha[1])*alpha[2],
                                (1-alpha[0])*(1-alpha[1])*(1-alpha[2]))

        g_z = theta - theta_expr

        # Switching function
        c = [ca.vertcat(q[1], v[1], v[0])]

        # dynamics
        f_x = ca.vertcat(theta[0]*v[0],
                         theta[0]*v[1],
                         a_n*(-mu*theta[1]+mu*theta[2]),
                         -g*theta[0] + a_n*theta[1],
                         theta[0])
    else:
        # multipliers
        alpha = ca.SX.sym('alpha', 2)
        theta = ca.SX.sym('theta', 2)
        z = theta
        z0 = np.array([0, 0])

        theta_expr = ca.vertcat((alpha[0]+alpha[1])-(alpha[0]*alpha[1]),
                                (1-alpha[0])*(1-alpha[1]))

        g_z = theta - theta_expr

        # Switching function
        c = [ca.vertcat(q[1], v[1])]

        # dynamics
        f_x = ca.vertcat(theta[0]*v[0],
                         theta[0]*v[1],
                         0,
                         -g*theta[0] + a_n*theta[1],
                         theta[0])

    model = ns.NosnocModel(x=ca.vertcat(x, t), f_x=[f_x], alpha=[alpha], c=c, x0=np.concatenate((x0, [0])), t_var=t,
                           z=z, z0=z0, g_z=g_z)
    return model


def get_default_options():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 1.
    opts.homotopy_update_rule = ns.HomotopyUpdateRule.LINEAR
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.cross_comp_mode = ns.CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER
    opts.mpcc_mode = ns.MpccMode.SCHOLTES_INEQ
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['print_level'] = 0

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    opts.speed_of_time_variables = ns.SpeedOfTimeVariableMode.LOCAL
    opts.time_freezing_tolerance = 0.0
    opts.N_finite_elements = 5
    opts.max_iter_homotopy = 9
    return opts


def solve_particle(opts=None, plot=True, friction=False, x0=np.array([0, 1, 3, 0]), Nsim=20, Tsim=2.5):
    if opts is None:
        opts = get_default_options()

    model = get_particle_model(opts, friction, x0)

    Tstep = Tsim / Nsim
    opts.terminal_time = Tstep

    solver = ns.NosnocSolver(opts, model)

    looper = ns.NosnocSimLooper(solver, np.concatenate((x0, [0.0])), Nsim)
    looper.run()
    results = looper.get_results()
    if plot:
        plot_results(results, opts, friction, Nsim)

    return results


def plot_results(results, opts, friction, Nsim):
    X_sim = np.array(results['X_sim'])
    theta_sim = np.array(results['z_sim'])
    n_theta = 3 if friction else 2
    theta_sim = theta_sim.reshape((opts.N_finite_elements*Nsim, n_theta))
    x = X_sim[:, 0]
    y = X_sim[:, 1]
    vx = X_sim[:, 2]
    vy = X_sim[:, 3]
    t = X_sim[:, -1]

    # Plot Trajectory
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(t, vx, label=r"$v_x$")
    plt.plot(t, vy, label=r"$v_y$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$v(t)$")
    plt.legend(loc='best', framealpha=0.1)
    plt.grid()
    plt.tight_layout()

    # Plot Multipliers
    plt.figure()
    plt.plot(t[1:], theta_sim[:, 0], label=r"$\theta_1$")
    plt.plot(t[1:], theta_sim[:, 1], label=r"$\theta_2$")
    if friction:
        plt.plot(t[1:], theta_sim[:, 2], label=r"$\theta_3$")
    plt.legend(loc='best', framealpha=0.1)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\theta(t)$")
    plt.grid()

    plt.show()


if __name__ == '__main__':
    opts = get_default_options()
    solve_particle(opts=opts, plot=True, friction=True)
