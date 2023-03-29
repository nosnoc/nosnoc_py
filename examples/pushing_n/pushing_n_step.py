import nosnoc as ns
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
from functools import partial

r1 = 0.1
r2 = 1

def get_pusher_ocp_step(opts, lift_algebraic, x_goal, multijump=False):
    # hopper model
    # model vars
    q = ca.SX.sym('q', 5)
    v = ca.SX.sym('v', 5)
    t = ca.SX.sym('t')
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', 2)
    sot = ca.SX.sym('sot')

    alpha = ca.SX.sym('alpha', 2)
    theta = ca.SX.sym('theta', 2)

    z = theta
    z0 = np.ones((2,))

    # dims
    n_q = 5
    n_v = 5
    n_x = n_q + n_v

    # state equations
    m1 = 1
    m2 = 2
    
    mu = 0.0   # friction coeficient
    I2 = 1/4*m2*r2**2;

    q1 = q[0:2]
    q2 = q[2:4]
    theta2 = q[4]
    v1 = v[0:2]
    v2 = v[2:4]
    omega = v[4]

    R_matrix = ca.vertcat(ca.horzcat(ca.cos(theta2), -ca.sin(theta2)),
                          ca.horzcat(ca.sin(theta2), ca.cos(theta2)));
    # inertia matrix
    M = np.diag([m1, m1, m2, m2, I2])

    cv = 2
    eps = 1e-1
    f_drag = cv*ca.vertcat(v1/ca.norm_2(v1+eps), v2/ca.norm_2(v2+eps));
    
    p = 6
    f_c = ca.sum1((R_matrix@(q1-q2))**p)-(r1+r2)**p
    #f_c_fun = ca.Function('f_c_fun', {x}, {f_c})

    f_v = ca.vertcat(ca.vertcat(u,ca.SX.zeros(2)) - f_drag, 0)

    f_c_normal = ca.jacobian(f_c, q)
    v_normal = f_c_normal@v

    # The control u[2] is a slack for modelling of nonslipping constraints.
    ubu = np.array([50, 50, 20])
    lbu = np.array([-50, -50, 0.1])
    u_guess = np.array([0, 0, 1])

    ubx = np.array([10, 10, 10, 10, 10, 5, 5, 5, 5, 5, np.inf])
    lbx = np.array([-10, -10, -10, -10, -10, -5, -5, -5, -5, -5, -np.inf])

    Q = np.diag([50, 50, 10])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    R = np.diag([0.1, 0.1])

    x0 = [0,0];
    x1 = [0,6];
    x2 = [3,0];
    x3 = [3,6];

    theta0 = [0];
    theta1 = [3*np.pi/4];
    theta2 = [0];
    theta3 = [0];

    x_kpts = np.array([x0, x1, x2, x3])
    theta_kpts = np.array([theta0, theta1, theta2, theta3])
    T_kpts = [0, 6, 6+3*np.sqrt(5), 12+3*np.sqrt(5)]
    
    x_interp = interp1d(T_kpts, x_kpts.T, 'linear')
    theta_interp = interp1d(T_kpts, theta_kpts.T, 'previous')
    x_track_ref = x_interp( np.linspace(0,12+3*np.sqrt(5), opts.N_stages+1))
    theta_track_ref = theta_interp(np.linspace(0,12+3*np.sqrt(5), opts.N_stages+1))

    x_ref = np.vstack((x_track_ref, theta_track_ref))
    
    # Hand create least squares cost
    p_x_ref = ca.SX.sym('x_ref', 3)

    ref = ca.vertcat(q2, theta2)
    f_q = sot*(ca.transpose(ref-p_x_ref)@Q@(ref-p_x_ref) + ca.transpose(u)@R@u)
    #f_q_T = ca.transpose(x - x_end)@Q_terminal@(x - x_end)

    # hand crafted time freezing :)
    a_n = 10
    J_normal = f_c_normal.T
    inv_M = ca.inv(M)
    f_ode = sot * ca.vertcat(v, inv_M@f_v, 1)
    f_aux = ca.vertcat(ca.SX.zeros(n_q, 1), inv_M@J_normal*a_n, 0)

    c = [ca.vertcat(f_c, v_normal)]
    f_x = theta[0]*f_ode + theta[1]*f_aux

    g_z = theta-ca.vertcat(alpha[0]+(1-alpha[0])*alpha[1],
                           (1-alpha[0])*(1-alpha[1]))

    x0 = [0,-1.5,0,0,-np.pi/6,0,0,0,0,0]
    model = ns.NosnocModel(x=ca.vertcat(x, t), f_x=[f_x], alpha=[alpha], c=c, x0=np.concatenate((x0, [0])),
                           u=ca.vertcat(u, sot), p_time_var=p_x_ref, p_time_var_val=x_ref[:, 1:].T, t_var=t,
                           z=z, z0=z0, g_z=g_z)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, lbx=lbx, ubx=ubx, u_guess=u_guess)

    v_normal_fun = ca.Function('v_normal_fun', [x], [v_normal])
    f_c_fun = ca.Function('f_c_fun', [x], [f_c])
    return model, ocp, x_ref, v_normal_fun, f_c_fun


def get_default_options_step():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-4
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 1.
    opts.homotopy_update_rule = ns.HomotopyUpdateRule.LINEAR
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.mpcc_mode = ns.MpccMode.SCHOLTES_INEQ
    opts.cross_comp_mode = ns.CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
    opts.print_level = 4

    opts.opts_casadi_nlp['ipopt']['max_iter'] = 10000
    opts.opts_casadi_nlp['ipopt']['acceptable_tol'] = 1e-6
    opts.opts_casadi_nlp['ipopt']['print_level'] = 5

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    opts.N_stages = 30
    opts.N_finite_elements = 2
    opts.max_iter_homotopy = 1
    return opts


def solve_ocp_step(opts=None, plot=True, lift_algebraic=False, x_goal=1.0, ref_as_init=False, multijump=False):
    if opts is None:
        opts = get_default_options_step()
        opts.terminal_time = 12+3*np.sqrt(5)
        opts.N_stages = 30

    model, ocp, x_ref, v_normal_fun, f_c_fun = get_pusher_ocp_step(opts, lift_algebraic, x_goal, multijump)

    solver = ns.NosnocSolver(opts, model, ocp)

    # Calculate time steps and initialize x to [xref, t]
    if ref_as_init:
        opts.initialization_strategy = ns.InitializationStrategy.EXTERNAL
        t_steps = np.linspace(0, opts.terminal_time, opts.N_stages+1)
        solver.set('x', np.c_[np.zeros((opts.N_stages, 2)),x_ref[:, 1:].T,np.zeros((opts.N_stages, 5)), t_steps[1:]])
    results = solver.solve()
    if plot:
        plot_results(results, opts, x_ref, v_normal_fun, f_c_fun)

    return results

def init_func(box_trail):
    box_trail.set_data([], [])

    return box_trail,

frame_cnt = 0

def animate_pusher(state, box, pusher, box_trail):
    global frame_cnt
    x_box, y_box = state[2], state[3]
    theta_box = state[4]
    x_pusher, y_pusher = state[0], state[1]
    box.set_xy((x_box-r2*np.cos(theta_box) + r2*np.sin(theta_box), y_box -r2*np.cos(theta_box) - r2*np.sin(theta_box)))
    box.rotation_point = 'center'
    box.set_angle(np.rad2deg(theta_box))
    
    pusher.center = x_pusher, y_pusher
    
    box_trail.set_data(np.append(box_trail.get_xdata(orig=False), x_box), np.append(box_trail.get_ydata(orig=False), y_box))
    #plt.savefig(str(frame_cnt)+'.pdf')
    # frame_cnt += 1
    return box, pusher, box_trail


def plot_results(results, opts, x_ref, v_normal_fun, f_c_fun):
    fig, ax = plt.subplots()

    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)

    state = results['x_traj'][0]
    x_box, y_box = state[2], state[3]
    theta_box = state[4]
    x_pusher, y_pusher = state[0], state[1]
    
    box = patches.Rectangle((-r2, -r2), 2*r2, 2*r2, angle=theta_box, animated=True, rotation_point='center', )    
    pusher = patches.Circle((0, 0), r1, color='black', animated=True)

    box.set_xy((x_box-r2*np.sin(theta_box), y_box - r2*np.cos(theta_box)))
    pusher.center = x_pusher, y_pusher
    
    ax.add_patch(box)
    ax.add_patch(pusher)
    ax.set_aspect('equal', 'box')
    ax.plot(x_ref[0, :], x_ref[1, :], color='lightgrey')
    
    box_trail, = ax.plot([], [], color='b', alpha=0.5)
    ani = FuncAnimation(fig, partial(animate_pusher, box=box, pusher=pusher, box_trail=box_trail),
                        init_func=partial(init_func, box_trail=box_trail),
                        frames=results['x_traj'], blit=True, repeat=False)
    try:
        ani.save('pusher.gif', writer='imagemagick', fps=10)
    except Exception:
        print("install imagemagick to save as gif")

    # Plot Trajectory
    # plt.figure()
    # x_traj = np.array(results['x_traj'])
    # t = x_traj[:, -1]
    # x = x_traj[:, 0]
    # y = x_traj[:, 1]
    # theta = x_traj[:, 2]
    # leg_len = x_traj[:, 3]
    # plt.subplot(4, 1, 1)
    # plt.plot(results['t_grid'], t)
    # plt.subplot(4, 1, 2)
    # plt.plot(results['t_grid'], x, color='r')
    # plt.plot(results['t_grid'], y, color='b')
    # plt.plot(results['t_grid_u'], x_ref[:, 0], color='r', alpha=0.5, linestyle='--')
    # plt.plot(results['t_grid_u'], x_ref[:, 1], color='b', alpha=0.5, linestyle='--')
    # plt.subplot(4, 1, 3)
    # plt.plot(results['t_grid'], theta, color='b')
    # plt.subplot(4, 1, 4)
    # plt.plot(results['t_grid'], leg_len, color='b')
    # plt.plot(results['t_grid_u'], x_ref[:, 3], color='b', alpha=0.5, linestyle='--')
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(results['t_grid'], f_c_fun(x_traj[:, :-1].T).full().T)
    # plt.subplot(3, 1, 2)
    # plt.plot(results['t_grid'], v_tangent_fun(x_traj[:, :-1].T).full().T)
    # plt.subplot(3, 1, 3)
    # plt.plot(results['t_grid'], v_normal_fun(x_traj[:, :-1].T).full().T)
    # # Plot Controls
    # plt.figure()
    # u_traj = np.array(results['u_traj'])
    # reaction = u_traj[:, 0]
    # leg_force = u_traj[:, 1]
    # slack = u_traj[:, 2]
    # sot = u_traj[:, 3]
    # plt.subplot(4, 1, 1)
    # plt.step(results['t_grid_u'], np.concatenate((reaction, [reaction[-1]])))
    # plt.subplot(4, 1, 2)
    # plt.step(results['t_grid_u'], np.concatenate((leg_force, [leg_force[-1]])))
    # plt.subplot(4, 1, 3)
    # plt.step(results['t_grid_u'], np.concatenate((slack, [slack[-1]])))
    # plt.subplot(4, 1, 4)
    # plt.step(results['t_grid_u'], np.concatenate((sot, [sot[-1]])))
    plt.show()

if __name__ == '__main__':
    results = solve_ocp_step()
    breakpoint()
