"""
Two discs that must swap positions while avoiding a central obstacle, solved as an optimal control
problem with the FESD-J discretization of a Complementarity Lagrangian System (CLS).

Only the first disc is actuated (a thrust force ``u``). The second disc can only be moved by making
inelastic contact with the first one. The optimizer discovers a trajectory in which the first disc
pushes the second around the obstacle so that both reach the other's initial position.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import nosnoc

#supress warning until new vdx release
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*__array_wrap__.*",
    category=DeprecationWarning,
)


# ------------------------------------------------------------------ problem data

# masses and radii of the two discs
M1, M2 = 2.0, 1.0
R1, R2 = 0.3, 0.2

# obstacle (a disc the two bodies must go around)
R_OB = 1.0
Q_OB = np.array([0.0, 0.0])

# initial and (swapped) target positions of the disc centers
Q10 = np.array([-2.0, 0.0])
Q20 = np.array([2.0, 0.0])
X0 = np.concatenate([Q10, Q20, np.zeros(4)])
X_REF = np.concatenate([Q20, Q10, np.zeros(4)])  # swap the two discs, at rest

# drag force parameters (regularized so the norm is smooth at zero velocity)
CV = 2.0
DRAG_EPS = 1e-1

# horizon and discretization
T = 4.0
N_STAGES = 50
N_FE = 2
N_S = 2

# bounds
UBX = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0])
UBU = np.array([30.0, 30.0])

# objective weights
Q_DIAG = np.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
R_DIAG = np.array([0.1, 0.1])
Q_TERMINAL_SCALE = 100.0


def get_disc_model():
    """Build the two-disc manipulation problem as a `nosnoc.model.Cls`."""
    q = ca.SX.sym("q", 4)
    v = ca.SX.sym("v", 4)
    u = ca.SX.sym("u", 2)
    x = ca.vertcat(q, v)

    q1, q2 = q[0:2], q[2:4]
    v1, v2 = v[0:2], v[2:4]

    # A velocity dependent drag acting on both discs, regularized at zero velocity.
    f_drag = CV*ca.vertcat(v1/ca.norm_2(v1 + DRAG_EPS), v2/ca.norm_2(v2 + DRAG_EPS))
    # Only the first disc is actuated.
    f_v = ca.vertcat(u, 0.0, 0.0) - f_drag

    # Single contact between the two discs: they may not interpenetrate.
    f_c = ca.norm_2(q1 - q2)**2 - (R1 + R2)**2

    # Obstacle avoidance as a path constraint g_path <= 0: each disc center must stay outside a
    # circle of radius (r_ob + r_i) around the obstacle.
    g_path = -ca.vertcat(
        ca.sumsqr(q1 - Q_OB) - (R_OB + R1)**2,
        ca.sumsqr(q2 - Q_OB) - (R_OB + R2)**2,
    )

    x_ref = ca.DM(X_REF)
    Q = ca.diag(ca.DM(Q_DIAG))
    R = ca.diag(ca.DM(R_DIAG))
    f_q = (x - x_ref).T@Q@(x - x_ref) + u.T@R@u
    f_q_T = Q_TERMINAL_SCALE*(x - x_ref).T@Q@(x - x_ref)

    return nosnoc.model.Cls(
        x=x,
        u=u,
        x0=X0,
        M=np.diag([M1, M1, M2, M2]),
        f_v=f_v,
        f_c=f_c,
        e=0.0,          # inelastic impacts
        mu=0.0,         # frictionless
        lbx=-UBX, ubx=UBX,
        lbu=-UBU, ubu=UBU,
        # g_path <= 0. The upper bound must be given explicitly, the model default is +inf.
        g_path=g_path, lbg_path=-np.inf*np.ones(2), ubg_path=np.zeros(2),
        f_q=f_q,
        f_q_T=f_q_T,
        name="disc_around_obstacle",
    )


def get_default_options(**kwargs):
    default_args = {
        "N_stages": N_STAGES,
        "N_finite_elements": N_FE,
        "n_s": N_S,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "dcs_mode": nosnoc.DcsMode.CLS,

        "use_fesd": True,
        #NOTE: this can be changed to nosnoc.ClsDiscretization.RELAXED_OC
        "cls_discretization": nosnoc.ClsDiscretization.FESD_J, 

        # Matches the MATLAB reference, which leaves cross_comp_mode at the default FE_STAGE.
        # For Patel OC use FE_FE 
        "cross_comp_mode": nosnoc.CrossComplementarityMode.STAGE_STAGE,
        "step_equilibration": nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED,
       
                 
        "g_path_at_fe": True,    # enforce the obstacle constraint at every finite element boundary
        "T": T,
        #"rho_h": 0.0, # no step equilibration, the finite element lengths are fixed
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_solver_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.homotopy_update_slope = 0.1
    solver_opts.sigma_0 = 1e1
    solver_opts.complementarity_tol = 1e-6
    solver_opts.N_homotopy = 100
    solver_opts.opts_casadi_nlp["ipopt"]["max_iter"] = 2000
    for k, val in kwargs.items():
        setattr(solver_opts, k, val)
    return solver_opts


def solve(opts=None, solver_opts=None):
    model = get_disc_model()
    if opts is None:
        opts = get_default_options()
    if solver_opts is None:
        solver_opts = get_default_solver_options()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)
    solver.solve()
    return solver


def _circle(radius, center):
    tt = np.linspace(0, 2*np.pi, 100)
    return radius*np.cos(tt) + center[0], radius*np.sin(tt) + center[1]


def animate(solver, save_path=None, show=True):
    """
    Animate the two discs moving around the obstacle, matching the MATLAB reference.

    If `save_path` is given the animation is written as a GIF (e.g.
    "discs_switch_position_obstacle.gif").
    """
    nosnoc.latexify_plot()
    x = solver.get("x")          # (N+1, n_x), states at the control grid points
    q1, q2 = x[:, 0:2], x[:, 2:4]
    n_frames = q1.shape[0]

    lim = np.max(np.abs(x[:, 0:4])) + 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid()

    # static elements: obstacle and the two (faded) target outlines
    ax.plot(*_circle(R_OB, Q_OB), "k-", lw=1.5)
    ax.plot(*_circle(R1, X_REF[0:2]), color="C0", alpha=0.4)
    ax.plot(*_circle(R2, X_REF[2:4]), color="C3", alpha=0.4)

    # animated disc outlines and their traced paths
    (disc1,) = ax.plot([], [], "-", color="C0", lw=2, label="disc 1")
    (disc2,) = ax.plot([], [], "-", color="C3", lw=2, label="disc 2")
    (trail1,) = ax.plot([], [], "-", color="C0", alpha=0.3)
    (trail2,) = ax.plot([], [], "-", color="C3", alpha=0.3)
    ax.legend(loc="upper right")

    def update(frame):
        disc1.set_data(*_circle(R1, q1[frame]))
        disc2.set_data(*_circle(R2, q2[frame]))
        trail1.set_data(q1[:frame+1, 0], q1[:frame+1, 1])
        trail2.set_data(q2[:frame+1, 0], q2[:frame+1, 1])
        return disc1, disc2, trail1, trail2

    # frame delay from the (fixed) step size, like the MATLAB DelayTime = h_k(1)
    h0 = T/(N_STAGES*N_FE)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000*h0, blit=True)

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=max(1, int(1/h0))))
        print(f"  saved animation to {save_path}")
    if show:
        plt.show()
    return anim


def plot_time_series(solver):
    """Velocities of both discs and the optimal control over time."""
    nosnoc.latexify_plot()
    x = solver.get("x")
    u = solver.get("u")
    t = solver.get_time_grid()
    t_u = solver.get_control_grid()

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(t, x[:, 4], label="$v_{1,x}$")
    axs[0].plot(t, x[:, 5], label="$v_{1,y}$")
    axs[0].set_ylabel("$v_1$")
    axs[0].legend(); axs[0].grid()
    axs[1].plot(t, x[:, 6], label="$v_{2,x}$")
    axs[1].plot(t, x[:, 7], label="$v_{2,y}$")
    axs[1].set_ylabel("$v_2$")
    axs[1].legend(); axs[1].grid()
    axs[2].step(t_u, np.vstack([u, u[-1]]), where="post")
    axs[2].set_ylabel("$u$")
    axs[2].set_xlabel("$t$ [s]")
    axs[2].legend(["$u_1$", "$u_2$"]); axs[2].grid()
    plt.tight_layout()
    plt.show()


def example(do_plot=True, save_gif=False):
    solver = solve()
    x = solver.get("x")
    q_end = x[-1, 0:4]
    print("disc around obstacle (CLS OCP)")
    print(f"  terminal position error {np.linalg.norm(q_end - X_REF[0:4]):.3e}")
    if do_plot:
        save_path = "discs_switch_position_obstacle.gif" if save_gif else None
        animate(solver, save_path=save_path)
        plot_time_series(solver)
    return solver


if __name__ == "__main__":
    example()
