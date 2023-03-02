## Hopper OCP
# example inspired by https://github.com/KY-Lin22/NIPOCPEC and https://github.com/thowell/motion_planning/blob/main/models/hopper.jl
# The methods and time-freezing refomulation are detailed in https://arxiv.org/abs/2111.06759


import nosnoc as ns
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def get_hopper_ocp_description(opts):
    ## Parameters
    v_slip_bound = 0.001
    x_goal = 0.7

    ## hopper model
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
    C = np.array([0,(mb + ml)*g,0,0]).T
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
    ubu= np.array([50, 50, 100, 20])
    lbu= np.array([-50, -50, 0, 1])

    ubx = np.array([x_goal+0.1, 1.5, np.pi, 0.50, 10, 10, 5, 5])
    lbx =  np.array([0, 0, -np.pi, 0.1, -10, -10, -5, -5])
    
    x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
    x_mid = np.array([(x_goal-0.1)/2+0.1, 0.8, 0, 0.1, 0, 0, 0, 0])
    x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])

    interpolator = CubicSpline([0, 0.5, 1], [x0, x_mid, x_end])
    x_ref = interpolator(np.linspace(0, 1, opts.N_stages))
    
    Q = np.diag([50, 50, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    
    u_ref = np.array([0, 0, 0])
    R = np.diag([0.01, 0.01, 1e-5])

    # path comp to avoid slipping
    g_comp_path = ca.horzcat(ca.vertcat(v_tangent, f_c), ca.vertcat(u[2], u[2]))

    # Hand create least squares cost
    p_x_ref = ca.SX.sym('x_ref', n_x)

    f_q = sot*(ca.transpose(x - p_x_ref)@Q@(x-p_x_ref) + ca.transpose(u)@R@u)
    f_q_T = ca.transpose(x - x_end)@Q_terminal@(x - x_end)

    ## hand crafted time freezing :)
    a_n = 100
    J_normal = f_c_normal
    J_tangent = f_c_tangent
    inv_M = ca.inv(M)
    f_ode = sot * ca.vertcat(v, inv_M@f_v, 1)
    
    inv_M_aux = inv_M
    inv_M_ext = np.eye(n_x+1)

    f_aux_pos = ca.vertcat(ca.SX.zeros(n_q, 1), inv_M_aux@(J_normal-J_tangent*mu)*a_n, 0)
    f_aux_neg = ca.vertcat(ca.SX.zeros(n_q, 1), inv_M_aux@(J_normal+J_tangent*mu)*a_n, 0)
    
    F = [ca.horzcat(f_ode, f_ode, f_aux_pos, f_aux_neg)]
    S = [np.array([[1,0,0],[-1,1,0],[-1,-1,1],[-1,-1,-1]])]
    c = [ca.vertcat(f_c, v_normal, v_tangent)]

    model = ns.NosnocModel(x=ca.vertcat(x,t), F=F, S=S, c=c, x0=np.concatenate((x0,[0])), u=ca.vertcat(u, sot), p_time_var=p_x_ref,p_time_var_val=x_ref, t_var=t)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_q_T, g_path_comp=g_comp_path)

    return model, ocp


def get_default_options():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-9
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_DELTA
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['max_iter'] = 1000
    opts.opts_casadi_nlp['ipopt']['acceptable_tol'] = 1e-6

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    opts.N_stages = 20
    opts.N_finite_elements = 3
    opts.max_iter_homotopy = 5
    return opts


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    [model, ocp] = get_hopper_ocp_description(opts)

    opts.terminal_time = 1

    solver = ns.NosnocSolver(opts, model, ocp)

    results = solver.solve()
    # print(f"{results['u_traj']=}")
    # print(f"{results['time_steps']=}")
    breakpoint()
    plot_results(results, opts)

    return results

def animate_robot(state):
    x_head, y_head = state[0], state[1]
    x_foot, y_foot = state[0] - state[3]*np.sin(state[2]), state[1] - state[3]*np.cos(state[2])
    head = plt.scatter(x_head, y_head, color='b', s=[100])
    foot = plt.scatter(x_foot, y_foot, color='r', s=[50])
    body, = plt.plot([x_foot,x_head], [y_foot, y_head], 'k')

    return head, foot, body
    

def plot_results(results, opts):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1,.9)
    ax.set_ylim(-0.1,1.5)
    patch = patches.Rectangle((-0.1,-0.1),1,0.1, color='grey')
    ax.add_patch(patch)
    # TODO: plot target trajectory
    ani = FuncAnimation(fig, animate_robot, frames=results['x_traj'], blit=True)

    plt.show()
    
    

solve_ocp()
