# Hopper OCP
# example inspired by https://github.com/KY-Lin22/NIPOCPEC and https://github.com/thowell/motion_planning/blob/main/models/hopper.jl
# The methods and time-freezing refomulation are detailed in https://arxiv.org/abs/2111.06759


import nosnoc as ns
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from functools import partial


DENSE = False
lift_algebraic = False
x_goal = 0.7


def get_hopper_ocp_description(opts):
    # hopper model
    # model vars
    q = ca.SX.sym('q', 4)
    v = ca.SX.sym('v', 4)
    t = ca.SX.sym('t')
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', 3)
    sot = ca.SX.sym('sot')

    alpha = ca.SX.sym('alpha', 3)
    theta = ca.SX.sym('theta', 3)
    if lift_algebraic:
        beta = ca.SX.sym('beta', 1)
        z = ca.vertcat(theta, beta)
        z0 = np.ones((4,))
    else:
        z = theta
        z0 = np.ones((3,))

    # dims
    n_q = 4
    n_v = 4
    n_x = n_q + n_v

    # state equations
    mb = 1      # body
    ml = 0.1    # link
    Ib = 0.25   # body
    Il = 0.025  # link
    mu = 0.45   # friction coeficient
    g = 9.81

    # inertia matrix
    M = np.diag([mb + ml, mb + ml, Ib + Il, ml])
    # coriolis and gravity
    C = np.array([0, (mb + ml)*g, 0, 0]).T
    # Control input matrix
    B = ca.vertcat(ca.horzcat(0, -np.sin(q[2])),
                   ca.horzcat(0, np.cos(q[2])),
                   ca.horzcat(1, 0),
                   ca.horzcat(0, 1))

    f_c_normal = ca.vertcat(0, 1, q[3]*ca.sin(q[2]), -ca.cos(q[2]))
    f_c_tangent = ca.vertcat(1, 0, q[3]*ca.cos(q[2]), ca.sin(q[2]))

    v_normal = f_c_normal.T@v
    v_tangent = f_c_tangent.T@v

    f_v = (-C + B@u[0:2])
    f_c = q[1] - q[3]*ca.cos(q[2])

    x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
    x_mid = np.array([(x_goal-0.1)/2+0.1, 0.8, 0, 0.1, 0, 0, 0, 0])
    x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])

    interpolator = CubicSpline([0, 0.5, 1], [x0, x_mid, x_end])
    x_ref = interpolator(np.linspace(0, 1, opts.N_stages))

    # The control u[2] is a slack for modelling of nonslipping constraints.
    ubu = np.array([50, 50, 100, 20])
    lbu = np.array([-50, -50, 0, 0.1])

    ubx = np.array([x_goal+0.1, 1.5, np.pi, 0.50, 10, 10, 5, 5, np.inf])
    lbx = np.array([0, 0, -np.pi, 0.1, -10, -10, -5, -5, -np.inf])

    Q = np.diag([50, 50, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    R = np.diag([0.01, 0.01, 100])

    # path comp to avoid slipping
    g_comp_path = ca.horzcat(v_tangent, theta[1]+theta[2])
    # Hand create least squares cost
    p_x_ref = ca.SX.sym('x_ref', n_x)

    f_q = sot*(ca.transpose(x - p_x_ref)@Q@(x-p_x_ref) + ca.transpose(u)@R@u)
    f_q_T = ca.transpose(x - x_end)@Q_terminal@(x - x_end)

    # hand crafted time freezing :)
    a_n = 100
    J_normal = f_c_normal
    J_tangent = f_c_tangent
    inv_M = ca.inv(M)
    f_ode = sot * ca.vertcat(v, inv_M@f_v, 1)

    inv_M_aux = inv_M

    f_aux_pos = ca.vertcat(ca.SX.zeros(n_q, 1), inv_M_aux@(J_normal-J_tangent*mu)*a_n, 0)
    f_aux_neg = ca.vertcat(ca.SX.zeros(n_q, 1), inv_M_aux@(J_normal+J_tangent*mu)*a_n, 0)

    c = [ca.vertcat(f_c, v_normal, v_tangent)]
    f_x = theta[0]*f_ode + theta[1]*f_aux_pos+theta[2]*f_aux_neg

    if lift_algebraic:
        g_z = ca.vertcat(theta-ca.vertcat(alpha[0]+beta,
                                          beta*alpha[2],
                                          beta*(1-alpha[2])),
                         beta-(1-alpha[0])*(1-alpha[1]))
    else:
        g_z = theta-ca.vertcat(alpha[0]+(1-alpha[0])*(1-alpha[1]),
                               (1-alpha[0])*(1-alpha[1])*alpha[2],
                               (1-alpha[0])*(1-alpha[1])*(1-alpha[2]))

    model = ns.NosnocModel(x=ca.vertcat(x, t), f_x=[f_x], alpha=[alpha], c=c, x0=np.concatenate((x0, [0])),
                           u=ca.vertcat(u, sot), p_time_var=p_x_ref, p_time_var_val=x_ref, t_var=t,
                           z=z, z0=z0, g_z=g_z)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_q_T, g_path_comp=g_comp_path, lbx=lbx, ubx=ubx)

    v_tangent_fun = ca.Function('v_normal_fun', [x], [v_tangent])
    v_normal_fun = ca.Function('v_normal_fun', [x], [v_normal])
    f_c_fun = ca.Function('f_c_fun', [x], [f_c])
    return model, ocp, x_ref, v_tangent_fun, v_normal_fun, f_c_fun


def get_default_options():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-9
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 10.
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['max_iter'] = 1000
    opts.opts_casadi_nlp['ipopt']['acceptable_tol'] = 1e-6

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    opts.N_stages = 20
    opts.N_finite_elements = 3
    opts.max_iter_homotopy = 6
    return opts


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    model, ocp, x_ref, v_tangent_fun, v_normal_fun, f_c_fun = get_hopper_ocp_description(opts)

    opts.terminal_time = 1

    solver = ns.NosnocSolver(opts, model, ocp)
    results = solver.solve()
    plot_results(results, opts, x_ref, v_tangent_fun, v_normal_fun, f_c_fun)

    return results


def init_func(htrail, ftrail):
    htrail.set_data([], [])
    ftrail.set_data([], [])

    return htrail, ftrail


def animate_robot(state, head, foot, body, ftrail, htrail):
    x_head, y_head = state[0], state[1]
    x_foot, y_foot = state[0] - state[3]*np.sin(state[2]), state[1] - state[3]*np.cos(state[2])
    head.set_offsets([x_head, y_head])
    foot.set_offsets([x_foot, y_foot])
    body.set_data([x_foot, x_head], [y_foot, y_head])

    ftrail.set_data(np.append(ftrail.get_xdata(orig=False), x_foot), np.append(ftrail.get_ydata(orig=False), y_foot))
    htrail.set_data(np.append(htrail.get_xdata(orig=False), x_head), np.append(htrail.get_ydata(orig=False), y_head))
    return head, foot, body, ftrail, htrail


def plot_results(results, opts, x_ref, v_tangent_fun, v_normal_fun, f_c_fun):
    fig, ax = plt.subplots()

    ax.set_xlim(0, x_goal+0.1)
    ax.set_ylim(-0.1, 1.1)
    patch = patches.Rectangle((-0.1, -0.1), x_goal+0.1, 0.1, color='grey')
    ax.add_patch(patch)
    ax.plot(x_ref[:, 0], x_ref[:, 1], color='lightgrey')
    head = ax.scatter([0], [0], color='b', s=[100])
    foot = ax.scatter([0], [0], color='r', s=[50])
    body, = ax.plot([], [], 'k')
    ftrail, = ax.plot([], [], color='r', alpha=0.5)
    htrail, = ax.plot([], [], color='b', alpha=0.5)
    ani = FuncAnimation(fig, partial(animate_robot, head=head, foot=foot, body=body, htrail=htrail, ftrail=ftrail),
                        init_func=partial(init_func, htrail=htrail, ftrail=ftrail),
                        frames=results['x_traj'], blit=True)
    try:
        ani.save('hopper.gif', writer='imagemagick', fps=10)
    except Exception:
        print("install imagemagick to save as gif")

    breakpoint()
    # Plot Trajectory
    plt.figure()
    x_traj = np.array(results['x_traj'])
    t = x_traj[:, -1]
    x = x_traj[:, 0]
    y = x_traj[:, 1]
    theta = x_traj[:, 2]
    leg_len = x_traj[:, 3]
    plt.subplot(4, 1, 1)
    plt.plot(results['t_grid'], t)
    plt.subplot(4, 1, 2)
    plt.plot(results['t_grid'], x)
    plt.plot(results['t_grid'], y)
    plt.subplot(4, 1, 3)
    plt.plot(results['t_grid'], theta)
    plt.subplot(4, 1, 4)
    plt.plot(results['t_grid'], leg_len)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(results['t_grid'], f_c_fun(x_traj[:, :-1].T).full().T)
    plt.subplot(3, 1, 2)
    plt.plot(results['t_grid'], v_tangent_fun(x_traj[:, :-1].T).full().T)
    plt.subplot(3, 1, 3)
    plt.plot(results['t_grid'], v_normal_fun(x_traj[:, :-1].T).full().T)
    # Plot Controls
    plt.figure()
    u_traj = np.array(results['u_traj'])
    reaction = u_traj[:, 0]
    leg_force = u_traj[:, 1]
    slack = u_traj[:, 2]
    sot = u_traj[:, 3]
    plt.subplot(4, 1, 1)
    plt.step(results['t_grid_u'], np.concatenate((reaction, [reaction[-1]])))
    plt.subplot(4, 1, 2)
    plt.step(results['t_grid_u'], np.concatenate((leg_force, [leg_force[-1]])))
    plt.subplot(4, 1, 3)
    plt.step(results['t_grid_u'], np.concatenate((slack, [slack[-1]])))
    plt.subplot(4, 1, 4)
    plt.step(results['t_grid_u'], np.concatenate((sot, [sot[-1]])))
    plt.show()


if __name__ == '__main__':
    solve_ocp()
