import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, horzcat, vertcat, cos, sin, inv
import nosnoc as ns

from pendulum_utils import plot_results

T_OCP = 1.0
N_STAGES = 10
T_STEP = T_OCP/N_STAGES

def get_default_opts(**kwargs):
    default_opts = {
        "rk_scheme"          : ns.RKScheme.RADAU_IIA,
        "n_s"                : 2,
        "N_stages"           : 25,  # number of control intervals
        "N_finite_elements"  : 1,   # number of finite element on every control intevral
        "T"                  : 5.0, # Time horizon
        "print_level"        : 0,
    }
    return ns.Options(**(default_opts | kwargs))


def cartpole_model():
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

    p_global = vertcat(x_ref, u_ref, m1, m2)

    p_global_val = np.concatenate([x_ref_val,u_ref_val,np.array([1.0, 0.1])])

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

    # specify initial and end state, cost ref and weight matrix
    x0 = np.array([1, 0 / 180 * np.pi, 0, 0])  # start downwards
    #x0 = np.array([0.0, 0 / 180 * np.pi, 0, 0])  # start downwards

    Q = np.diag([1, 10, 1, 1])
    Q_terminal = np.diag([1000, 1000, 10, 10])
    R = 0.1

    # Stage cost
    f_q = 0.5*((x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref))
    # terminal cost
    f_terminal = 0.5*((x - x_ref).T @ Q_terminal @ (x - x_ref))

    # bounds
    ubx = np.array([5.0, np.inf, np.inf, np.inf])
    lbx = -np.array([5.0, np.inf, np.inf, np.inf])

    u_max = 20.0
    lbu = -np.array([u_max])
    ubu = np.array([u_max])

    model = ns.model.Dae(
        x=x,
        f_x=vertcat(v, inv(M) @ (f_all)),
        x0=x0,
        u=u,
        p_global=p_global,
        p_global_val=p_global_val,
        lbu=lbu,
        ubu=ubu,
        f_q=f_q,
        f_q_T=f_terminal,
        lbx=lbx,
        ubx=ubx,
    )
    return model

def run_example(**kwargs):
    opts = get_default_opts(**kwargs)
    model = cartpole_model()
    ipopt_opts = {"ipopt.linear_solver" : "mumps"}

    solver = ns.OcpSolver(model,opts,ipopt_opts)
    solver.solve()

    return solver

if __name__ == "__main__":
    solver = run_example()
    plot_results(solver)
    ns.dump_hp_functions(solver.model, solver.opts, "cartpole")
    breakpoint()
    
