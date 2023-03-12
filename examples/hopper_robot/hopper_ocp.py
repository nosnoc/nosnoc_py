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

LONG = True
DENSE = False


def get_hopper_ocp_description(opts):
    # Parameters
    x_goal = 0.7

    # hopper model
    # model vars
    q = ca.SX.sym('q', 4)
    v = ca.SX.sym('v', 4)
    t = ca.SX.sym('t')
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', 3)
    sot = ca.SX.sym('sot')

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

    # The control u[2] is a slack for modelling of nonslipping constraints.
    ubu = np.array([50, 50, 100, 20])
    lbu = np.array([-50, -50, 0, 1])

    ubx = np.array([x_goal+0.1, 1.5, np.pi, 0.50, 10, 10, 5, 5])
    lbx = np.array([0, 0, -np.pi, 0.1, -10, -10, -5, -5])

    if LONG:
        x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
        x_mid1 = np.array([0.4, 0.65, 0, 0.2, 0, 0, 0, 0])
        x_mid2 = np.array([0.6, 0.5, 0, 0.5, 0, 0, 0, 0])
        x_mid3 = np.array([0.9, 0.65, 0, 0.2, 0, 0, 0, 0])
        x_end = np.array([1.3, 0.5, 0, 0.5, 0, 0, 0, 0])

        interpolator1 = CubicSpline([0, 0.5, 1], [x0, x_mid1, x_mid2])
        interpolator2 = CubicSpline([0, 0.5, 1], [x_mid2, x_mid3, x_end])

        x_ref1 = interpolator1(np.linspace(0, 1, int(np.floor(opts.N_stages/2))))
        x_ref2 = interpolator2(np.linspace(0, 1, int(np.floor(opts.N_stages/2))))
        x_ref = np.concatenate([x_ref1, x_ref2])
    else:
        x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
        x_mid = np.array([(x_goal-0.1)/2+0.1, 0.8, 0, 0.1, 0, 0, 0, 0])
        x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])

        interpolator = CubicSpline([0, 0.5, 1], [x0, x_mid, x_end])
        x_ref = interpolator(np.linspace(0, 1, opts.N_stages))

    Q = np.diag([50, 50, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    R = np.diag([0.01, 0.01, 1e-5])

    # path comp to avoid slipping
    g_comp_path = ca.horzcat(ca.vertcat(v_tangent, f_c), ca.vertcat(u[2], u[2]))

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

    if DENSE:
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
                           u=ca.vertcat(u, sot), p_time_var=p_x_ref, p_time_var_val=x_ref, t_var=t)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_q_T, g_path_comp=g_comp_path, lbx=lbx, ubx=ubx)

    return model, ocp, x_ref


def get_default_options():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-9
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 100.
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_DELTA
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['max_iter'] = 1000
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


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    [model, ocp, x_ref] = get_hopper_ocp_description(opts)

    opts.terminal_time = 1

    solver = ns.NosnocSolver(opts, model, ocp)

    results = solver.solve()
    plot_results(results, opts, x_ref)

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


def plot_results(results, opts, x_ref):
    fig, ax = plt.subplots()
    if LONG:
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.1, 1.1)
        patch = patches.Rectangle((-0.1, -0.1), 1.6, 0.1, color='grey')
        ax.add_patch(patch)
    else:
        ax.set_xlim(0, .8)
        ax.set_ylim(-0.1, 1.1)
        patch = patches.Rectangle((-0.1, -0.1), 1, 0.1, color='grey')
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
    plt.show()


if __name__ == '__main__':
    solve_ocp()
