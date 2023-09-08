# Ball in box ocp

import nosnoc as ns
import casadi as ca
import numpy as np
import tk
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from functools import partial

def get_ball_ocp(omega, opts):
    # model params
    N_periods = 2
    alpha0 = np.pi/4 # inital angle
    opts.terminal_time = N_periods*(2*np.pi/abs(omega))
    ## Model Parameters
    time_var_reference = 1
    qx_c = 0.0
    qy_c = 0.0
    v_target = 5
    R = 1
    friction_exists = 0
    u_max_R = 5e1 # amplitude of control force vector

    # objective parameters
    rho_q = 1
    rho_v = 1
    rho_u = 0
    ## Inital Value
    qx0 = R*np.sin(alpha0)
    qy0 = R*np.cos(alpha0)
    vx0 = R*omega*np.cos(alpha0)
    vy0 = -R*omega*np.sin(alpha0)
    t0  = 0
    x0 = np.array([qx0, qy0, vx0, vy0, t0])
    ## Model parameters for time freezing
    k_tf = 100   # stiffness
    gamma_tf = 1 # restitution coefficient
    c_tf = 2*abs(np.log(gamma_tf))*((k_tf) /(np.pi**2+np.log(gamma_tf)**2) )**(1/2)
    T_res = 2*np.pi/np.sqrt((4*k_tf-c_tf**2))
    
    # model vars
    qx = ca.SX.sym('qx')
    qy = ca.SX.sym('qy')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    t = ca.SX.sym('t')
    q = ca.vertcat(qx,qy)
    v = ca.vertcat(vx,vy)
    x = ca.vertcat(q,v,t)
    ## control
    ux = ca.SX.sym('ux')
    uy = ca.SX.sym('uy')
    u = ca.vertcat(ux,uy)
    u0 = np.array([0,0])

    n_x = 5
    n_q = 2
    n_u = 2

    umax = np.inf
    lbx = -np.inf*np.ones(n_x)
    ubx = np.inf*np.ones(n_x)
    lbu = -umax*np.ones(n_u)
    ubu = umax*np.ones(n_u)

    unit_size = 0.05*R*1
    b_bottom = -(R+1*unit_size)
    a_right = (R+2*unit_size)
    b_top = (R+3*unit_size)
    a_left = -(R+4*unit_size)
    
    # every constraint funcion corresponds to a simplex (note that the c_i might be vector valued)
    c_1 = qy-b_bottom; # bottom
    c_2 = -qx+a_right; # right
    c_3 = -qy+b_top; # top
    c_4 = qx-a_left; # left
    # sign matrix for the modes
    S = np.array([[1, 1, 1, 1], # interior
                  [-1, 1, 1, 1],  # bottom
                  [-1, -1, 1, 1],  # bottom right
                  [1, -1, 1, 1],  # right
                  [1, -1, -1, 1],  # top right
                  [1, 1, -1, 1],  # top
                  [1, 1, -1, -1],  # top left
                  [1, 1, 1, -1],  # left
                  [-1, 1, 1, -1]]); # bottom left
    c = ca.vertcat(c_1,c_2,c_3,c_4);

    ## auxiliary dynamics
    f_aux_right = ca.vertcat(vx,0,-k_tf*(qx-a_right)-c_tf*vx,0,0)
    f_aux_bottom = ca.vertcat(0,vy,0,-k_tf*(qy-b_bottom)-c_tf*vy,0)
    f_aux_left = ca.vertcat(vx,0,-k_tf*(qx-a_left)-c_tf*vx,0,0)
    f_aux_top = ca.vertcat(0,vy,0,-k_tf*(qy-b_top)-c_tf*vy,0)
    f_ode = ca.vertcat(vx,vy,ux,uy,1)
    f_zero = np.zeros((n_x,1))

    f_11 = f_ode
    f_12 = f_aux_bottom
    f_13 = f_aux_bottom+f_aux_right
    f_14 = f_aux_right
    f_15 = f_aux_top+f_aux_right
    f_16 = f_aux_top
    f_17 = f_aux_top+f_aux_left
    f_18 = f_aux_left
    f_19 = f_aux_bottom+f_aux_left
    # in matrix form
    F = ca.horzcat(f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19)

    ## objective
    qx_ref = R*ca.sin(omega*t+alpha0)
    qy_ref = R*ca.cos(omega*t+alpha0);    
    vx_ref = R*omega*ca.cos(omega*t+alpha0)
    vy_ref = -R*omega*ca.sin(omega*t+alpha0)
    q_ref = ca.vertcat(qx_ref,qy_ref)
    v_ref = ca.vertcat(vx_ref,vy_ref)
    f_q = rho_q*ca.transpose(q-q_ref)@(q-q_ref)+rho_v*ca.transpose(v-v_ref)@(v-v_ref)+rho_u*ca.transpose(u)@u; 
    
    # Terminal Cost
    f_q_T = 0
    
    ##  general nonlinear constinrst
    g_path = ca.transpose(u)@u
    g_path_lb = np.array([-np.inf])
    g_path_ub = np.array([u_max_R**2])
    
    model = ns.NosnocModel(x=x, F=[F], c=[c], S=[S], x0=x0,
                           u=u, t_var=t)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, u_guess=u0,
                       f_q=f_q, f_terminal=f_q_T,
                       lbx=lbx, ubx=ubx,
                       g_path=g_path, lbg=g_path_lb, ubg=g_path_ub
                       )
    return model, ocp


def get_default_options_step():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEWART
    opts.comp_tol = 1e-6
    opts.use_fesd = True
    opts.homotopy_update_slope = 0.1
    opts.sigma_0 = 1.
    opts.homotopy_update_rule = ns.HomotopyUpdateRule.LINEAR
    opts.n_s = 2
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.cross_comp_mode = ns.CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
    opts.mpcc_mode = ns.MpccMode.ELASTIC_INEQ
    opts.speed_of_time_variables = ns.SpeedOfTimeVariableMode.LOCAL
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['print_level'] = 5
    opts.opts_casadi_nlp['ipopt']['tol'] = 1e-7
    opts.opts_casadi_nlp['ipopt']['max_iter'] = 1000

    opts.time_freezing = True
    opts.equidistant_control_grid = True
    opts.N_stages = 40
    opts.N_finite_elements = 4
    opts.max_iter_homotopy = 2
    return opts


def solve_ocp_step(opts=None, plot=True):
    if opts is None:
        opts = get_default_options_step()
        opts.terminal_time = 3.0
        opts.N_stages = 30

    omega = -3*np.pi
    model, ocp = get_ball_ocp(omega, opts)

    solver = ns.NosnocSolver(opts, model, ocp)
    # for python3
    import sys
    with open('test.txt', 'w') as w:
        std = sys.stdout
        sys.stdout = w
        solver.print_problem()
        sys.stdout = std
    results = solver.solve()
    breakpoint()
    if plot:
        plot_results(results, opts, omega)

    return results




def plot_results(results, opts, omega):
    x_traj = np.array(results['x_traj'])
    u_traj = np.array(results['u_traj'])
    t = x_traj[:, -1]
    x = x_traj[:, 0]
    y = x_traj[:, 1]
    vx = x_traj[:, 2]
    vy = x_traj[:, 3]
    ux = u_traj[:, 0]
    uy = u_traj[:, 1]
    
    # Plot Trajectory
    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(t,x)
    plt.plot(t,y)
    plt.subplot(1,2,2)
    plt.plot(t,vx)
    plt.plot(t,vy)

    # Plot Controls
    plt.figure()
    plt.plot(results['t_grid_u'], np.concatenate((ux, [ux[-1]])))
    plt.plot(results['t_grid_u'], np.concatenate((uy, [uy[-1]])))

    plt.show()


if __name__ == '__main__':
    opts = get_default_options_step()
    opts.N_stages = 40
    opts.time_freezing_tolerance = 0.0
    solve_ocp_step(opts=opts, plot=True)
