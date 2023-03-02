## Hopper OCP
# example inspired by https://github.com/KY-Lin22/NIPOCPEC and https://github.com/thowell/motion_planning/blob/main/models/hopper.jl
# The methods and time-freezing refomulation are detailed in https://arxiv.org/abs/2111.06759


import nosnoc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def get_hopper_ocp_description():
    ## Parameters
    v_slip_bound = 0.001
    x_goal = 0.7

    ## hopper model
    # model vars
    q = ca.SX.sym('q', 4)
    v = ca.SX.sym('v', 4)
    x = ca.vertcat(q,v)
    u = ca.SX.sym('u', 3)

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
    B = np.array([[0, -np.sin(q(3))],
                  [0, np.cos(q(3))],
                  [1, 0],
                  [0, 1]])

    f_c_normal = ca.vertcat(0, 1, q[3]*ca.sin(q[2]), -ca.cos(q[2]))
    f_c_tangent = ca.vertcat(1, 0, q[3]*ca.cos(q[2]), ca.sin(q[2]))

    v_normal = f_c_normal.T*v
    v_tangent = f_c_tangent.T*v

    f_v = -C + B@u[1:2]
    f_c = q[1] - q[3]*ca.cos(q[2])

    # The control u(3) is a slack for modelling of nonslipping constraints.
    ubu= np.array([50, 50, 100])
    lbu= np.array([-50, -50, 0])

    ubx = np.array([x_goal+0.1, 1.5, pi, 0.50, 10, 10, 5, 5])
    lbx =  np.array([0, 0, -pi, 0.1, -10, -10, -5, -5])
    
    x0 = np.array([0.1, 0.5, 0, 0.5, 0, 0, 0, 0])
    x_mid = np.array([(x_goal-0.1)/2+0.1, 0.8, 0, 0.1, 0, 0, 0, 0])
    x_end = np.array([x_goal, 0.5, 0, 0.5, 0, 0, 0, 0])
    
    Q = np.diag([50, 50, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([50, 50, 50, 50, 0.1, 0.1, 0.1, 0.1])

    Q = np.diag([50, 50, 20, 50, 0.1, 0.1, 0.1, 0.1])
    Q_terminal = np.diag([300, 300, 300, 300, 0.1, 0.1, 0.1, 0.1])
    
    u_ref = np.array([0, 0, 0])
    R = np.diag([0.01, 0.01, 1e-5])

    # path comp to avoid slipping
    g_comp_path = ca.horzcat(ca.vertcat(v_tangent, f_c), ca.vertcat(u[2]. u[2]))

    ## hand crafted time freezing :)
    # TODO: bruh


                 
def get_default_options():
    opts = nosnoc.NosnocOpts()
    # opts.pss_mode = nosnoc.PssMode.STEP
    opts.use_fesd = True
    comp_tol = 1e-6
    opts.comp_tol = comp_tol
    opts.homotopy_update_slope = 0.1
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED
    opts.print_level = 1

    opts.N_stages = 20
    opts.N_finite_elements = 3
    return opts


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    [model, ocp] = get_hopper_ocp_description()

    opts.terminal_time = 0.08

    solver = nosnoc.NosnocSolver(opts, model, ocp)

    results = solver.solve()
    # print(f"{results['u_traj']=}")
    # print(f"{results['time_steps']=}")

    return results
