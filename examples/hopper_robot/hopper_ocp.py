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

LONG = False
DENSE = True
HEIGHT = 1.0


def get_hopper_ocp_description(opts, x_goal, dense, multijump=False):
    # hopper model
    # model vars
    q = ca.SX.sym('q', 4)
    v = ca.SX.sym('v', 4)
    t = ca.SX.sym('t')
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', 3)
    sot = ca.SX.sym('sot')

    theta = ca.SX.sym('theta', 8)

    # dims
    n_q = 4
    n_v = 4
    n_x = n_q + n_v

    # state equations
    mb = 1      # body
    ml = 0.1    # link
    Ib = 0.25   # body
    Il = 0.025  # link
    mu = 0.45   # friction coefficient
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

    if LONG:
        x_goal = 1.3
        x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
        x_mid1 = np.array([0.4, 0.65, 0, 0.2, 0, 0, 0, 0])
        x_mid2 = np.array([0.6, 0.5, 0, 0.5, 0, 0, 0, 0])
        x_mid3 = np.array([0.9, 0.65, 0, 0.2, 0, 0, 0, 0])
        x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])

        interpolator1 = CubicSpline([0, 0.5, 1], [x0, x_mid1, x_mid2])
        interpolator2 = CubicSpline([0, 0.5, 1], [x_mid2, x_mid3, x_end])

        x_ref1 = interpolator1(np.linspace(0, 1, int(np.floor(opts.N_stages/2))))
        x_ref2 = interpolator2(np.linspace(0, 1, int(np.floor(opts.N_stages/2))))
        x_ref = np.concatenate([x_ref1, x_ref2])
    else:
        if multijump:
            n_jumps = int(np.round(x_goal-0.1))
            x_step = (x_goal-0.1)/n_jumps
            x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
            x_ref = np.empty((0, n_x))
            x_start = x0
            # TODO: if N_stages not divisible by n_jumps then this doesn't work... oh well
            for ii in range(n_jumps):
                # Parameters
                x_mid = np.array([x_start[0] + x_step/2, HEIGHT, 0, 0.1, 0, 0, 0, 0])
                x_end = np.array([x_start[0] + x_step, 0.5, 0, 0.5, 0, 0, 0, 0])
                interpolator = CubicSpline([0, 0.5, 1], [x_start, x_mid, x_end])
                t_pts = np.linspace(0, 1, int(np.floor(opts.N_stages/n_jumps))+1)
                x_ref = np.concatenate((x_ref, interpolator(t_pts[:-1])))
                x_start = x_end
            x_ref = np.concatenate((x_ref, np.expand_dims(x_end, axis=0)))
        else:
            x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
            x_mid = np.array([(x_goal-0.1)/2+0.1, HEIGHT, 0, 0.1, 0, 0, 0, 0])
            x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])

            interpolator = CubicSpline([0, 0.5, 1], [x0, x_mid, x_end])
            x_ref = interpolator(np.linspace(0, 1, opts.N_stages+1))
    # The control u[2] is a slack for modelling of nonslipping constraints.
    ubu = np.array([50, 50, 100, 20])
    lbu = np.array([-50, -50, 0, 0.1])
    u_guess = np.array([0, 0, 0, 1])

    ubx = np.array([x_goal+0.1, 1.5, np.pi, 0.50, 10, 10, 5, 5, np.inf])
    lbx = np.array([0, 0, -np.pi, 0.1, -10, -10, -5, -5, -np.inf])

    Q = np.diag([100, 100, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    R = np.diag([0.01, 0.01, 1e-5])

    # path comp to avoid slipping
    g_comp_path = ca.horzcat(ca.vertcat(v_tangent, -v_tangent), ca.vertcat(theta[-1]+theta[-2], theta[-1]+theta[-2]))

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

    if dense:
        F = [ca.horzcat(f_ode, f_ode, f_ode, f_ode, f_ode, f_ode, f_aux_pos, f_aux_neg)]
        S = [np.array([[1, 1, 1],
                       [1, 1, -1],
                       [1, -1, 1],
                       [1, -1, -1],
                       [-1, 1, 1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [-1, -1, -1]])]
    else:
        F = [ca.horzcat(f_ode, f_ode, f_aux_pos, f_aux_neg)]
        S = [np.array([[1, 0, 0], [-1, 1, 0], [-1, -1, 1], [-1, -1, -1]])]
    c = [ca.vertcat(f_c, v_normal, v_tangent)]

    model = ns.NosnocModel(x=ca.vertcat(x, t), F=F, S=S, c=c, x0=np.concatenate((x0, [0])),
                           u=ca.vertcat(u, sot), p_time_var=p_x_ref, p_time_var_val=x_ref[1:, :], t_var=t, theta=[theta])
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, u_guess=u_guess, f_q=f_q, f_terminal=f_q_T, g_path_comp=g_comp_path, lbx=lbx, ubx=ubx)

    v_tangent_fun = ca.Function('v_normal_fun', [x], [v_tangent])
    v_normal_fun = ca.Function('v_normal_fun', [x], [v_normal])
    f_c_fun = ca.Function('f_c_fun', [x], [f_c])
    return model, ocp, x_ref, v_tangent_fun, v_normal_fun, f_c_fun


def get_default_options():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEWART
    opts.use_fesd = True
    comp_tol = 1e-9
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 100.
    opts.homotopy_update_rule = ns.HomotopyUpdateRule.LINEAR
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.mpcc_mode = ns.MpccMode.SCHOLTES_INEQ
    #opts.cross_comp_mode = ns.CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['max_iter'] = 4000
    opts.opts_casadi_nlp['ipopt']['acceptable_tol'] = 1e-6

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    if LONG:
        opts.N_stages = 30
    else:
        opts.N_stages = 20
    opts.N_finite_elements = 3
    opts.max_iter_homotopy = 6
    return opts


def solve_ocp(opts=None, plot=True, dense=DENSE, ref_as_init=False, x_goal=1.0, multijump=False):
    if opts is None:
        opts = get_default_options()
        opts.terminal_time = 5.0
        opts.N_stages = 50

    model, ocp, x_ref, v_tangent_fun, v_normal_fun, f_c_fun = get_hopper_ocp_description(opts, x_goal, dense, multijump)

    solver = ns.NosnocSolver(opts, model, ocp)

    # Calculate time steps and initialize x to [xref, t]
    if ref_as_init:
        opts.initialization_strategy = ns.InitializationStrategy.EXTERNAL
        t_steps = np.linspace(0, opts.terminal_time, opts.N_stages)
        solver.set('x', np.c_[x_ref[1:, :], t_steps])

    results = solver.solve()
    if plot:
        plot_results(results, opts, x_ref, v_tangent_fun, v_normal_fun, f_c_fun, x_goal)

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


def plot_results(results, opts, x_ref, v_tangent_fun, v_normal_fun, f_c_fun, x_goal):
    fig, ax = plt.subplots()
    if LONG:
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.1, 1.1)
        patch = patches.Rectangle((-0.1, -0.1), 1.6, 0.1, color='grey')
        ax.add_patch(patch)
    else:
        ax.set_xlim(0, x_goal+0.1)
        ax.set_ylim(-0.1, HEIGHT+0.5)
        patch = patches.Rectangle((-0.1, -0.1), x_goal+0.2, 0.1, color='grey')
        ax.add_patch(patch)
    ax.plot(x_ref[:, 0], x_ref[:, 1], color='lightgrey')
    head = ax.scatter([0], [0], color='b', s=[100])
    foot = ax.scatter([0], [0], color='r', s=[50])
    body, = ax.plot([], [], 'k')
    ftrail, = ax.plot([], [], color='r', alpha=0.5)
    htrail, = ax.plot([], [], color='b', alpha=0.5)
    ani = FuncAnimation(fig, partial(animate_robot, head=head, foot=foot, body=body, htrail=htrail, ftrail=ftrail),
                        init_func=partial(init_func, htrail=htrail, ftrail=ftrail),
                        frames=results['x_traj'], blit=True, repeat=False)
    try:
        ani.save('hopper.gif', writer='imagemagick', fps=10)
    except Exception:
        print("install imagemagick to save as gif")

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
    solve_ocp(x_goal=5.0, multijump=True, ref_as_init=True)
