"""
Gearbox example with multiple modes.

Extension of the original model with two modes to three modes. The modes are
given by two auxillary variables and one switching function. The voronoi-regions
are thus given in a 3D space. The hysteresis curves can overlap in this
3D space and are solved faster than the 2D version.
"""

import pickle
import nosnoc
import casadi as ca
import numpy as np
from math import ceil, log
import matplotlib.pyplot as plt
from enum import Enum
from nosnoc.plot_utils import plot_voronoi_2d, plot_colored_line_3d

# Hystheresis parameters
v1 = 10
v2 = 14

# Model parameters
q_goal = 150
v_goal = 0
v_max = 25
u_max = 5

# fuel costs:
C = [1, 1.8, 2.5, 3.2]
# ratios
n = [1, 2, 3, 4]


def calc_dist(a, b):
    """Calculate distance."""
    print(np.norm_2(a - b))


class ZMode(Enum):
    """Z Mode."""

    TYPE_1_0 = 1
    TYPE_2_0 = 2
    PAPER_TYPE_2 = 3


def create_options():
    """Create nosnoc options."""
    opts = nosnoc.NosnocOpts()
    opts.print_level = 2
    # Degree of interpolating polynomial
    opts.n_s = 3
    # === MPCC settings ===
    # upper bound for elastic variables
    opts.s_elastic_max = 1e1
    # in penalty methods  1: J = J+(1/p)*J_comp (direct)  , 0 : J = p*J+J_comp (inverse)
    opts.objective_scaling_direct = 0
    # === Penalty/Relaxation paraemetr ===
    # initial smoothing parameter
    opts.sigma_0 = 1.0
    # end smoothing parameter
    opts.sigma_N = 1e-2  # 1e-10
    # decrease rate
    opts.homotopy_update_slope = 0.1
    # number of steps
    opts.N_homotopy = ceil(abs(
        log(opts.sigma_N / opts.sigma_0) / log(opts.homotopy_update_slope))) + 1
    opts.comp_tol = 1e-14

    # IPOPT Settings
    opts.nlp_max_iter = 5000

    # New setting: time freezing settings
    opts.pss_mode = nosnoc.PssMode.STEWART
    opts.mpcc_mode = nosnoc.MpccMode.SCHOLTES_INEQ
    return opts


def push_equation(a_push, psi, zero_point):
    """Eval push equation."""
    multip = 1
    return a_push * multip * (psi - zero_point) ** 2 / (1 + multip * (psi - zero_point)**2) + 1e-6


def create_gearbox_voronoi(use_simulation=False, q_goal=None, traject=None,
                           use_traject=False, use_traject_constraint=True,
                           shift=0.5, mode=ZMode.PAPER_TYPE_2):
    """Create a gearbox."""
    if not use_traject and q_goal is None:
        raise Exception("You should provide a traject or a q_goal")

    # State variables:
    q = ca.SX.sym("q")  # position
    v = ca.SX.sym("v")  # velocity
    L = ca.SX.sym("L")  # Fuel usage
    w1 = ca.SX.sym('w1')  # Auxillary variable
    w2 = ca.SX.sym('w2')  # Auxillary variable
    t = ca.SX.sym('t')  # Time variable
    X = ca.vertcat(q, v, L, w1, w2, t)
    X0 = np.array([0, 0, 0, 0, 0, 0]).T
    lbx = np.array([0, 0, -ca.inf, 0, 0, 0]).T
    ubx = np.array([ca.inf, v_max, ca.inf, 2, 2, ca.inf]).T

    if use_traject:
        p_traj = ca.SX.sym('traject')
    else:
        p_traj = ca.SX.sym('dummy', 0, 1)

    # Controls
    if not use_simulation:
        u = ca.SX.sym('u')  # drive
        s = ca.SX.sym('s')  # Length of time
        U = ca.vertcat(u)
        lbu = np.array([-u_max])
        ubu = np.array([u_max])
    else:
        u = ca.SX.sym('u')  # drive
        s = 1
        lbu = u
        ubu = u
        U = [u]

    # Tracking gearbox:
    psi = (v-v1)/(v2-v1)
    z = ca.vertcat(psi, w1, w2)
    a = 1/4
    b = 1/4
    if mode == ZMode.TYPE_1_0:
        Z = [
            np.array([b, -a, 0]),
            np.array([b,  a, 0]),
            np.array([1-b, 1-a, 0]),
            np.array([1-b, 1+a, 0])
        ]
    elif mode == ZMode.TYPE_2_0:
        Z = [
            np.array([b,  -a, 0]),
            np.array([b,   a, 0]),
            np.array([1-b, 1-a, 0]),
            np.array([1-b, 1+a, 0]),

            np.array([b + shift, 1,  -a]),
            np.array([b + shift, 1,   a]),
            np.array([1-b + shift, 1, 1-a]),
            np.array([1-b + shift, 1, 1+a])
        ]
    elif mode == ZMode.PAPER_TYPE_2:
        # Shift can be 0.5 or 2
        Z = [
            np.array([b,  -a, 0]),
            np.array([b,   a, 0]),
            np.array([1-b, 1-a, 0]),
            np.array([1-b, 1+a, 0]),

            np.array([b + shift, 1,  -a]),
            np.array([b + shift, 1,   a]),
            np.array([1 - b + shift, 1, 1-a]),
            np.array([1 - b + shift, 1, 1+a]),

            np.array([b + 2 * shift, 1-a,  1]),
            np.array([b + 2 * shift, 1+a,  1]),
            np.array([1 - b + 2 * shift, 2-a, 1]),
            np.array([1 - b + 2 * shift, 2+a, 1])
        ]

    g_ind = [ca.vertcat(*[
        ca.norm_2(z - zi)**2 for zi in Z
    ])]

    # Traject
    f_q = 0
    g_path = 0
    if use_traject:
        if use_traject_constraint:
            print("use trajectory as constraint")
            g_path = p_traj - q
        else:
            print("use trajectory as cost")
            f_q = 0.001 * (p_traj - q)**2

        g_terminal = ca.vertcat(q-p_traj, v-v_goal)
    else:
        g_terminal = ca.vertcat(q-q_goal, v-v_goal)

    f_terminal = t

    # System dynamics
    f_A = ca.vertcat(
        v, n[0]*u, C[0], 0, 0, 1
    )
    f_B = ca.vertcat(
        v, n[1]*u, C[1], 0, 0, 1
    )
    f_C = ca.vertcat(
        v, n[2]*u, C[2], 0, 0, 1
    )
    f_D = ca.vertcat(
        v, n[3]*u, C[3], 0, 0, 1
    )

    a_push = 2
    push_down_eq = push_equation(-a_push, psi, 1)
    push_up_eq = push_equation(a_push, psi, 0)

    f_push_down_w1 = ca.vertcat(0, 0, 0, push_down_eq, 0, 0)
    f_push_up_w1 = ca.vertcat(0, 0, 0, push_up_eq, 0, 0)

    if mode == ZMode.TYPE_1_0:
        f_1 = [
            s * (2 * f_A - f_push_down_w1),
            s * (f_push_down_w1),
            s * (f_push_up_w1),
            s * (2 * f_B - f_push_up_w1)
        ]
    elif mode == ZMode.TYPE_2_0:
        push_down_eq = push_equation(-a_push, psi, 1 + shift)
        push_up_eq = push_equation(a_push, psi, 0 + shift)
        f_push_down_w2 = ca.vertcat(0, 0, 0, 0, push_down_eq, 0)
        f_push_up_w2 = ca.vertcat(0, 0, 0, 0, push_up_eq, 0)
        f_1 = [
            s * (2 * f_A - f_push_down_w1),
            s * (f_push_down_w1),
            s * (f_push_up_w1),
            s * (2 * f_B - f_push_up_w1),
            s * (2 * f_B - f_push_down_w2),
            s * (f_push_down_w2),
            s * (f_push_up_w2),
            s * (2 * f_C - f_push_up_w2),
        ]
    elif mode == ZMode.PAPER_TYPE_2:
        push_down_eq = push_equation(-a_push, psi, 1 + shift)
        push_up_eq = push_equation(a_push, psi, 0 + shift)
        f_push_down_w2 = ca.vertcat(0, 0, 0, 0, push_down_eq, 0)
        f_push_up_w2 = ca.vertcat(0, 0, 0, 0, push_up_eq, 0)

        push_down_eq = push_equation(-a_push, psi, 1 + 2 * shift)
        push_up_eq = push_equation(a_push, psi, 0 + 2 * shift)
        f_push_down_w3 = ca.vertcat(0, 0, 0, push_down_eq, 0, 0)
        f_push_up_w3 = ca.vertcat(0, 0, 0, push_up_eq, 0, 0)
        f_1 = [
            s * (2 * f_A - f_push_down_w1),
            s * (f_push_down_w1),
            s * (f_push_up_w1),
            s * (2 * f_B - f_push_up_w1),
            s * (2 * f_B - f_push_down_w2),
            s * (f_push_down_w2),
            s * (f_push_up_w2),
            s * (2 * f_C - f_push_up_w2),
            s * (2 * f_C - f_push_down_w3),
            s * (f_push_down_w3),
            s * (f_push_up_w3),
            s * (2 * f_D - f_push_up_w3),
        ]

    F = [ca.horzcat(*f_1)]

    if not use_simulation:
        model = nosnoc.NosnocModel(
            x=X, F=F, g_Stewart=g_ind, x0=X0, u=U, t_var=t,
            p_time_var=p_traj,
            p_time_var_val=traject, v_global=s,
            name="gearbox"
        )
    else:
        model = nosnoc.NosnocModel(
            x=X, F=F, g_Stewart=g_ind, x0=X0, t_var=t,
            p_global=u, p_global_val=np.array([0]),
            name="gearbox"
        )
    return model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal, Z


def plot(x_list, t_grid, u_list, t_grid_u, Z):
    """Plot."""
    q = [x[0] for x in x_list]
    v = [x[1] for x in x_list]
    w1 = [x[-3] for x in x_list]
    w2 = [x[-2] for x in x_list]
    aux = [x[-2] + x[-3] for x in x_list]
    t = [x[-1] for x in x_list]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t_grid, x_list, label=[
        "$q$ (position)", "$v$ (speed)", "$L$ (cost)",
        "$w$ (auxillary variable)", "$w_2$", "$t$ (time)"
    ])
    plt.xlabel("Simulation Time [$s$]")
    plt.legend()
    if u_list is not None:
        plt.subplot(1, 3, 2)
        plt.plot(t_grid_u[:-1], u_list, label=["u", "s"])
        plt.xlabel("Simulation Time [$s$]")
        plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(t, q, label="Position vs actual time")
    plt.xlabel("Actual Time [$s$]")

    Z_2d = [[z[0], z[1] + z[2]] for z in Z]
    psi = [(vi - v1) / (v2 - v1) for vi in v]
    ax = plot_voronoi_2d(Z_2d, show=False)
    im = ax.scatter(psi, aux, c=t_grid, cmap=plt.hot())
    im.set_label('Time')
    plt.colorbar(im, ax=ax)
    plt.xlabel("$\\psi(x)$")
    plt.ylabel("$w$")

    plot_colored_line_3d(psi, w1, w2, t)
    plt.show()


def simulation(Tsim=6, Nsim=30, with_plot=True, shift=0.5, mode=ZMode.PAPER_TYPE_2):
    """Simulate the temperature control system with a fixed input."""
    opts = create_options()
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal, Z = create_gearbox_voronoi(
        use_simulation=True, q_goal=q_goal, mode=mode, shift=shift
    )
    Tstep = Tsim / Nsim
    opts.N_finite_elements = 2
    opts.N_stages = 1
    opts.terminal_time = Tstep
    opts.sigma_N = 1e-2

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, p_values=np.array([[
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        10, 10, 10, 10, 10, 10, -10, -10, -10, -10,
        -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
    ]]).T)
    looper.run()
    results = looper.get_results()
    print(f"Ends in zone: {np.argmax(results['theta_sim'][-1][-1])}")
    print(results['theta_sim'][-1][-1])
    plot(results["X_sim"], results["t_grid"], None, None, Z=Z)


def custom_callback(prob, w_opt, lambda0, x0, iteration):
    """Make feasible."""
    if iteration < 1:
        return w_opt

    ind_x_all = nosnoc.utils.flatten_outer_layers(prob.ind_x, 2)
    x_sol = w_opt[ind_x_all]

    w1 = x_sol[:, -3]
    w2 = x_sol[:, -2]
    w = w1 + w2

    # Bring w to the plane:
    w = np.clip(w, 0, 6)
    w1 = w // 2 + np.clip(w % 2, 0, 1)
    w2 = w // 2 + np.clip(w % 2 - 1, 0, 1)

    x_sol[:, 1] = w1
    x_sol[:, 2] = w2
    w_opt[ind_x_all] = x_sol

    return w_opt


def control(shift=0.5, mode=ZMode.PAPER_TYPE_2):
    """Execute one Control step."""
    N = 10
    # traject = np.array([[q_goal * (i + 1) / N for i in range(N)]]).T
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal, Z = create_gearbox_voronoi(
        q_goal=q_goal, shift=shift, mode=mode
    )
    opts = create_options()
    opts.N_finite_elements = 3
    opts.n_s = 2
    opts.N_stages = N
    opts.terminal_time = 10
    opts.initialization_strategy = nosnoc.InitializationStrategy.EXTERNAL

    ocp = nosnoc.NosnocOcp(
        lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_terminal,
        g_terminal=g_terminal,
        lbv_global=np.array([0.1]), ubv_global=np.array([1e3]),
        lbx=lbx, ubx=ubx
    )
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    # solver.callback = custom_callback
    solver.set('v_global', np.array([1]))
    solver.set('x', np.vstack((
        np.linspace(0, q_goal, N),
        np.linspace(0, v_goal, N),
        np.zeros((3, N)),
        np.ones((1, N))
    )).T)
    solver.set('u', np.vstack((
        np.zeros((1, N)),
    )).T)
    results = solver.solve()
    solver.model.w0 = results['w_sol']
    plot(
        results["x_traj"], results["t_grid"],
        results["u_list"], results["t_grid_u"], Z=Z
    )
    with open("data_3d.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    from sys import argv
    if len(argv) == 2:
        print("TYPE 2")
        control(shift=float(argv[1]), mode=ZMode.TYPE_2_0)
    elif len(argv) > 2:
        print("TYPE 3")
        control(shift=float(argv[1]), mode=ZMode.PAPER_TYPE_2)
    else:
        control()
