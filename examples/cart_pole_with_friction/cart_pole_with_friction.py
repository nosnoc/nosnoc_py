import numpy as np
import casadi as ca
from pendulum_utils import plot_results

import nosnoc


def get_cart_pole_model_and_ocp(F_friction: float=2.0, use_fillipov: bool = True):
    # symbolics
    px = ca.SX.sym('px')
    theta = ca.SX.sym('theta')
    q = ca.vertcat(px, theta)

    v = ca.SX.sym('v')
    theta_dot = ca.SX.sym('theta_dot')
    q_dot = ca.vertcat(v, theta_dot)

    x = ca.vertcat(q, q_dot)
    u = ca.SX.sym('u')  # control

    m1 = 1  # cart
    m2 = 0.1  # link
    link_length = 1
    g = 9.81
    # Inertia matrix
    M = ca.vertcat(ca.horzcat(m1 + m2, m2 * link_length * ca.cos(theta)),
                ca.horzcat(m2 * link_length * ca.cos(theta), m2 * link_length**2))
    # Coriolis force
    C = ca.SX.zeros(2, 2)
    C[0, 1] = -m2 * link_length * theta_dot * ca.sin(theta)

    # all forces = Gravity + Control + Coriolis; Note: friction added later
    f_all = ca.vertcat(u, -m2 * g * link_length * ca.sin(theta)) - C @ q_dot

    x0 = np.array([1, 0 / 180 * np.pi, 0, 0])  # start downwards

    if use_fillipov:
        # Dynamics with $ v > 0$
        f_1 = ca.vertcat(q_dot, ca.inv(M) @ (f_all - ca.vertcat(F_friction, 0)))
        # Dynamics with $ v < 0$
        f_2 = ca.vertcat(q_dot, ca.inv(M) @ (f_all + ca.vertcat(F_friction, 0)))

        F = [ca.horzcat(f_1, f_2)]
        # switching function (cart velocity)
        c = [v]
        # Sign matrix # f_1 for c=v>0, f_2 for c=v<0
        S = [np.array([[1], [-1]])]
        model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=x0, u=u)
    else:
        sigma = ca.SX.sym('sigma')

        f_ode = ca.vertcat(q_dot, ca.inv(M) @ (f_all - ca.vertcat( F_friction* ca.tanh(v/ sigma), 0)))
        model = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_ode, x0=x0, u=u, p=sigma)

    ## OCP description
    # cost
    x_ref = np.array([0, 180 / 180 * np.pi, 0, 0])  # end upwards
    u_ref = 0.0
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

    ocp = nosnoc.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_terminal, lbx=lbx, ubx=ubx)
    return model, ocp


def solve_example():
    # opts
    opts = nosnoc.NosnocOpts()
    opts.irk_scheme = nosnoc.IrkSchemes.RADAU_IIA
    opts.n_s = 2
    # opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_DELTA

    opts.N_stages = 20  # number of control intervals
    opts.N_finite_elements = 2  # number of finite element on every control intevral
    opts.terminal_time = 5.0  # Time horizon
    opts.print_level = 1

    model, ocp = get_cart_pole_model_and_ocp()

    ## Solve OCP
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    solver.problem.print()

    results = solver.solve()
    return results

def main():
    results = solve_example()
    plot_results(results)


if __name__ == "__main__":
    main()
