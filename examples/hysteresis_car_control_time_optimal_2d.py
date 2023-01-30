"""
Gearbox example with multiple modes.

Extension of the original model with two modes to three modes. The modes are still
given using one auxillary variable and one switching function. The voronoi regions
are thus given in a 2D space. It can easily be extended to N modes but the
hysteresis curves may not overlap.
"""


import nosnoc
import casadi as ca
import numpy as np
from math import ceil, log
import matplotlib.pyplot as plt
from enum import Enum

# Hystheresis parameters
v1 = 5
v2 = 10

# Model parameters
q_goal = 150
v_goal = 0
v_max = 30
u_max = 5

# fuel costs:
C = [1, 1.8, 2.5]
# ratios
n = [1, 2, 3]


class Stages(Enum):
    """Z Mode."""

    STAGE_1 = 1
    STAGE_2 = 2


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
    opts.sigma_0 = 1e1
    # end smoothing parameter
    opts.sigma_N = 1e-3  # 1e-10
    # decrease rate
    opts.homotopy_update_slope = 0.1
    # number of steps
    opts.N_homotopy = ceil(abs(
        log(opts.sigma_N / opts.sigma_0) / log(opts.homotopy_update_slope))) + 1
    opts.comp_tol = 1e-14

    # IPOPT Settings
    opts.nlp_max_iter = 500

    # New setting: time freezing settings
    opts.initial_theta = 0.5
    opts.time_freezing = False
    opts.pss_mode = nosnoc.PssMode.STEWART
    return opts


def push_equation(a_push, psi, zero_point):
    """Eval push equation."""
    return a_push * (psi - zero_point) ** 2 / (1 + (psi - zero_point)**2)


def create_gearbox_voronoi(u=None, q_goal=None, mode=Stages.STAGE_2,
                           psi_shift_2=0.6):
    """Create a gearbox."""
    # State variables:
    q = ca.SX.sym("q")  # position
    v = ca.SX.sym("v")  # velocity
    L = ca.SX.sym("L")  # Fuel usage
    w = ca.SX.sym('w')  # Auxillary variable
    t = ca.SX.sym('t')  # Time variable
    X = ca.vertcat(q, v, L, w, t)
    X0 = np.array([0, 0, 0, 0, 0]).T
    lbx = np.array([-ca.inf, 0, -ca.inf, -1, 0]).T
    ubx = np.array([ca.inf, v_max, ca.inf, 2, ca.inf]).T

    # Controls
    if u is None:
        u = ca.SX.sym('u')  # drive
        s = ca.SX.sym('s')  # Length of time
        U = ca.vertcat(u, s)
        lbu = np.array([-u_max, 0.5])
        ubu = np.array([u_max, 20])
    else:
        s = 1
        lbu = u
        ubu = u
        U = [u, s]

    # Tracking gearbox:
    psi = (v-v1)/(v2-v1)
    z = ca.vertcat(psi, w)
    if mode == Stages.STAGE_1:
        Z = [
            np.array([1 / 4, -1 / 4]),
            np.array([1 / 4, 1 / 4]),
            np.array([3 / 4, 3 / 4]),
            np.array([3 / 4, 5 / 4])
        ]
    elif mode == Stages.STAGE_2:
        if psi_shift_2 <= 1.0:
            print("Due to overlapping hysteresis curves, "
                  "this method might give a wrong result!")
        if psi_shift_2 <= 0.51:
            raise Exception("Regions overlap and this method will fail")

        Z = [
            np.array([1/4, -1/4]),  # Original mode 1
            np.array([1/4, 1/4]),  # Original mode 2
            np.array([3/4, 3/4]),  # Original mode 3
            np.array([3/4, 5/4]),  # Original mode 4

            np.array([psi_shift_2 + 1/4, 1 + -1/4]),  # Similar to mode 1
            np.array([psi_shift_2 + 1/4, 1 + 1/4]),  # Similar to mode 2
            np.array([psi_shift_2 + 3/4, 1 + 3/4]),  # Similar to mode 3
            np.array([psi_shift_2 + 3/4, 1 + 5/4]),  # Similar to mode 4
        ]

    g_ind = [ca.vertcat(*[
        ca.norm_2(z - zi)**2 for zi in Z
    ])]

    # Traject
    f_q = 0
    g_path = 0
    g_terminal = ca.vertcat(q-q_goal, v-v_goal)
    f_terminal = t

    # System dynamics
    f_A = ca.vertcat(
        v, n[0]*u, C[0], 0, 1
    )
    f_B = ca.vertcat(
        v, n[1]*u, C[1], 0, 1
    )
    f_C = ca.vertcat(
        v, n[2]*u, C[2], 0, 1
    )

    a_push = 4
    push_down_eq = push_equation(-a_push, psi, 1)
    push_up_eq = push_equation(a_push, psi, 0)

    f_push_down = ca.vertcat(0, 0, 0, push_down_eq, 0)
    f_push_up = ca.vertcat(0, 0, 0, push_up_eq, 0)

    if mode == Stages.STAGE_1:
        f_1 = [
            s * (2 * f_A - f_push_down),
            s * (f_push_down),
            s * (f_push_up),
            s * (2 * f_B - f_push_up)
        ]
    elif mode == Stages.STAGE_2:
        push_down_eq = push_equation(-a_push, psi, 1+psi_shift_2)
        push_up_eq = push_equation(a_push, psi, psi_shift_2)
        f_push_up_1 = ca.vertcat(0, 0, 0, push_up_eq, 0)
        f_push_down_1 = ca.vertcat(0, 0, 0, push_down_eq, 0)

        f_1 = [
            s * (2 * f_A - f_push_down),
            s * (f_push_down),
            s * (f_push_up),
            s * (2 * f_B - f_push_up),
            s * (2 * f_B - f_push_down_1),
            s * (f_push_down_1),
            s * (f_push_up_1),
            s * (2 * f_C - f_push_up_1),
        ]

    F = [ca.horzcat(*f_1)]

    if isinstance(U, ca.SX):
        model = nosnoc.NosnocModel(
            x=X, F=F, g_Stewart=g_ind, x0=X0, u=U, t_var=t,
            name="gearbox"
        )
    else:
        model = nosnoc.NosnocModel(
            x=X, F=F, g_Stewart=g_ind, x0=X0, t_var=t,
            name="gearbox"
        )
    return model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal


def plot(x_list, t_grid, u_list, t_grid_u):
    """Plot."""
    q = [x[0] for x in x_list]
    v = [x[1] for x in x_list]
    aux = [x[-2] for x in x_list]
    t = [x[-1] for x in x_list]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t_grid, x_list, label=[
        "$q$ (position)", "$v$ (speed)", "$L$ (cost)",
        "$w$ (auxillary variable)", "$t$ (time)"
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

    plt.figure()
    psi = [(vi - v1) / (v2 - v1) for vi in v]
    im = plt.scatter(psi, aux, c=t_grid, cmap=plt.hot())
    im.set_label('Time')
    plt.colorbar(im)
    plt.xlabel("$\\psi(x)$")
    plt.ylabel("$w$")
    plt.show()


def simulation(u=25, Tsim=6, Nsim=30, with_plot=True):
    """Simulate the temperature control system with a fixed input."""
    opts = create_options()
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal = create_gearbox_voronoi(u=u, q_goal=q_goal)
    Tstep = Tsim / Nsim
    opts.N_finite_elements = 2
    opts.N_stages = 1
    opts.terminal_time = Tstep
    opts.sigma_N = 1e-2

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    print(f"Ends in zone: {np.argmax(results['theta_sim'][-1][-1])}")
    print(results['theta_sim'][-1][-1])
    plot(results["X_sim"], results["t_grid"], None, None)


def control():
    """Execute one Control step."""
    N = 15
    model, lbx, ubx, lbu, ubu, f_q, f_terminal, g_path, g_terminal = create_gearbox_voronoi(
        q_goal=q_goal,
    )
    opts = create_options()
    opts.N_finite_elements = 6
    opts.n_s = 3
    opts.N_stages = N
    opts.terminal_time = 30
    opts.time_freezing = False
    opts.time_freezing_tolerance = 0.1
    opts.nlp_max_iter = 10000

    ocp = nosnoc.NosnocOcp(
        lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_terminal,
        g_terminal=g_terminal,
        lbx=lbx, ubx=ubx
    )
    solver = nosnoc.NosnocSolver(opts, model, ocp)
    results = solver.solve()
    plot(
        results["x_traj"], results["t_grid"],
        results["u_list"], results["t_grid_u"]
    )


if __name__ == "__main__":
    control()
