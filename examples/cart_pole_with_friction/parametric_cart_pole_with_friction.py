import numpy as np
from casadi import SX, horzcat, vertcat, cos, sin, inv
import nosnoc

def solve_paramteric_example(with_global_var=False):
    # options
    opts = nosnoc.NosnocOpts()
    opts.irk_scheme = nosnoc.IrkSchemes.RADAU_IIA
    opts.n_s = 2
    # opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_DELTA

    opts.N_stages = 20  # number of control intervals
    opts.N_finite_elements = 2  # number of finite element on every control intevral
    opts.terminal_time = 5.0  # Time horizon
    opts.print_level = 1

    ## Model defintion
    q = SX.sym('q', 2)
    v = SX.sym('v', 2)
    x = vertcat(q, v)
    u = SX.sym('u')  # control

    ## parametric version:
    # masses
    m1 = SX.sym('m1')  # cart
    m2 = SX.sym('m2')  # link
    x_ref = SX.sym('x_ref', 4)
    u_ref = SX.sym('u_ref', 1)
    x_ref_val = np.array([0, 180 / 180 * np.pi, 0, 0])  # end upwards
    u_ref_val = np.array([0.0])

    p_time_var = vertcat(x_ref, u_ref)
    p_time_var_val = np.tile(np.concatenate((x_ref_val, u_ref_val)), (opts.N_stages, 1))

    if with_global_var:
        p_global = vertcat(m2)
        p_global_val = np.array([0.1])

        v_global = m1
        lbv_global = np.array([1.0])
        ubv_global = np.array([100.0])
        v_global_guess = np.array([1.2])
    else:
        p_global = vertcat(m1, m2)
        p_global_val = np.array([1.0, 0.1])
        v_global = SX.sym("v_global", 0, 1)
        lbv_global = np.array([])
        ubv_global = np.array([])
        v_global_guess = np.array([])
    # actually vary x_ref theta entry over time
    # p_ind_theta = 1
    # p_time_var_val[:, p_ind_theta] = np.linspace(0.0, np.pi, opts.N_stages)

    link_length = 1
    g = 9.81
    # Inertia matrix
    M = vertcat(horzcat(m1 + m2, m2 * link_length * cos(q[1])),
                horzcat(m2 * link_length * cos(q[1]), m2 * link_length**2))
    # Coriolis force
    C = SX.zeros(2, 2)
    C[0, 1] = -m2 * link_length * v[1] * sin(q[1])

    # all forces = Gravity+Control+Coriolis (+Friction)
    f_all = vertcat(u, -m2 * g * link_length * sin(x[1])) - C @ v

    # friction between cart and ground
    F_friction = 2
    # Dynamics with $ v > 0$
    f_1 = vertcat(v, inv(M) @ (f_all - vertcat(F_friction, 0)))
    # Dynamics with $ v < 0$
    f_2 = vertcat(v, inv(M) @ (f_all + vertcat(F_friction, 0)))

    F = [horzcat(f_1, f_2)]
    # switching function (cart velocity)
    c = [v[0]]
    # Sign matrix # f_1 for c=v>0, f_2 for c=v<0
    S = [np.array([[1], [-1]])]

    # specify initial and end state, cost ref and weight matrix
    x0 = np.array([1, 0 / 180 * np.pi, 0, 0])  # start downwards

    model = nosnoc.NosnocModel(x=x,
                               F=F,
                               S=S,
                               c=c,
                               x0=x0,
                               u=u,
                               p_global=p_global,
                               p_global_val=p_global_val,
                               p_time_var=p_time_var,
                               v_global=v_global)

    Q = np.diag([10, 100, 1, 1])
    Q_terminal = np.diag([500, 100, 10, 10])
    R = 1.0

    # Stage cost
    f_q = (model.x - x_ref).T @ Q @ (model.x - x_ref) + (model.u - u_ref).T @ R @ (model.u - u_ref)
    # terminal cost
    f_terminal = (model.x - x_ref).T @ Q_terminal @ (model.x - x_ref)

    # bounds
    ubx = np.array([5.0, np.inf, np.inf, np.inf])
    lbx = -np.array([5.0, np.inf, np.inf, np.inf])

    u_max = 30.0
    lbu = -np.array([u_max])
    ubu = np.array([u_max])

    ocp = nosnoc.NosnocOcp(lbu=lbu,
                           ubu=ubu,
                           f_q=f_q,
                           f_terminal=f_terminal,
                           lbx=lbx,
                           ubx=ubx,
                           lbv_global=lbv_global,
                           ubv_global=ubv_global,
                           v_global_guess=v_global_guess)

    # create solver
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    # set / update parameters
    solver.set('p_time_var', p_time_var_val)
    solver.set('p_global', p_global_val)

    # solve OCP
    results = solver.solve()
    return results
