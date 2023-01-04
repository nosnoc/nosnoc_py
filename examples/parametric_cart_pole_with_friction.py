import numpy as np
from casadi import SX, horzcat, vertcat, cos, sin, inv
import matplotlib.pyplot as plt

import nosnoc

# TODO:% 2 ) make the masses m_1 and m_2 parameteres (constant over time)

PARAMETRIC = 1
def main():
    # opts
    opts = nosnoc.NosnocOpts()
    opts.irk_scheme = nosnoc.IrkSchemes.RADAU_IIA
    opts.n_s = 2
    opts.homotopy_update_rule = nosnoc.HomotopyUpdateRule.SUPERLINEAR
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    opts.equidistant_control_grid = True

    opts.N_stages = 50  # number of control intervals
    opts.N_finite_elements = 2  # number of finite element on every control intevral
    opts.terminal_time = 4.0  # Time horizon
    opts.print_level = 1
    # faster options
    opts.N_stages = 15  # number of control intervals

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
    p = vertcat(m1, m2, x_ref)

    x_ref_val = np.array([0, 180 / 180 * np.pi, 0, 0])  # end upwards

    # either provide single parameter vector
    p_val = np.concatenate((np.array([1.0, 0.1]), x_ref_val))
    # or matrix (one parameter vector for each control stage)
    p_val = np.tile(p_val, (opts.N_stages, 1))

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

    Q = np.diag([1.0, 100.0, 1.0, 1.0])
    Q_terminal = np.diag([100.0, 100.0, 10.0, 10.0])
    R = 1.0

    # bounds
    ubx = np.array([5.0, 240 / 180 * np.pi, 20.0, 20.0])
    lbx = np.array([-0.0, -240 / 180 * np.pi, -20.0, -20.0])
    u_max = 30.0
    u_ref = 0.0

    # Stage cost
    f_q = (x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref)
    # terminal cost
    f_q_T = (x - x_ref).T @ Q_terminal @ (x - x_ref)
    g_terminal = []

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0, u=u, p=p, p_val=p_val)

    lbu = -np.array([u_max])
    ubu = np.array([u_max])

    ocp = nosnoc.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_q_T=f_q_T, g_terminal=g_terminal,
                           lbx=lbx, ubx=ubx)

    ## Solve OCP
    solver = nosnoc.NosnocSolver(opts, model, ocp)

    results = solver.solve()
    # import pdb; pdb.set_trace()
    plot_results(results)


def plot_results(results):
    nosnoc.latexify_plot()

    x_traj = np.array(results["x_traj"])

    plt.figure()
    # states
    plt.subplot(3, 1, 1)
    plt.plot(results["t_grid"], x_traj[:, 0], label='$q_1$ - cart')
    plt.plot(results["t_grid"], x_traj[:, 1], label='$q_2$ - pole')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(results["t_grid"], x_traj[:, 2], label='$v_1$ - cart')
    plt.plot(results["t_grid"], x_traj[:, 3], label='$v_2$ - pole')
    plt.legend()
    plt.grid()

    # controls
    plt.subplot(3, 1, 3)
    plt.step(results["t_grid_u"], [results["u_traj"][0]] + results["u_traj"], label='u')

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
