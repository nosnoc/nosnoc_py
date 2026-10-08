
#Minimal Pds OCP: ball 1 (velocity controlled) pushes ball 2 to a target position.

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nosnoc

R = 0.2
X0 = np.array([-1.0, 0.0, 0.0, 0.0])
P2_REF = np.array([1.0, 0.5])
T = 4.0
N_STAGES = 30


def push_balls_model():
    p1 = ca.SX.sym("p1", 2)
    p2 = ca.SX.sym("p2", 2)
    u = ca.SX.sym("u", 2)
    x = ca.vertcat(p1, p2)

    model = nosnoc.model.Pds(
        x=x,
        u=u,
        x0=X0,
        f=ca.vertcat(u, 0, 0),
        f_c=ca.sumsqr(p1 - p2) - (2*R)**2,
        lbu=-np.ones(2), ubu=np.ones(2),
        f_q=ca.sumsqr(p2 - P2_REF) + 1e-2*ca.sumsqr(u),
        f_q_T=100*ca.sumsqr(p2 - P2_REF),
    )
    return model


def get_default_options(**kwargs) -> nosnoc.Options:
    default_args = {
        "N_stages": N_STAGES,
        "N_finite_elements": 1,
        "n_s": 1,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": False,
        "T": T,
    }
    return nosnoc.Options(**(default_args | kwargs))


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions() #you can of course adjust the settings of the reg homotopy
    model = push_balls_model()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    solver.solve()

    return solver


def plot_results(solver):
    """Positions of both balls, the controls, and the contact force with the ball distance."""
    x_res = solver.get("x")
    u_res = np.reshape(solver.get("u"), (N_STAGES, -1))
    lambda_res = np.ravel(solver.get_full("lambda_normal"))  # one value per stage (n_s = 1, N_FE = 1)
    t_grid = np.linspace(0, T, N_STAGES+1)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axs[0, 0].plot(t_grid, x_res[:, 0], label="$x$")
    axs[0, 0].plot(t_grid, x_res[:, 1], label="$y$")
    axs[0, 0].set_title("ball 1 position")
    axs[0, 1].plot(t_grid, x_res[:, 2], label="$x$")
    axs[0, 1].plot(t_grid, x_res[:, 3], label="$y$")
    axs[0, 1].axhline(P2_REF[0], color="C0", ls="--", alpha=0.5)
    axs[0, 1].axhline(P2_REF[1], color="C1", ls="--", alpha=0.5)
    axs[0, 1].set_title("ball 2 position (dashed: target)")
    axs[1, 0].stairs(u_res[:, 0], t_grid, label="$u_x$")
    axs[1, 0].stairs(u_res[:, 1], t_grid, label="$u_y$")
    axs[1, 0].set_title("controls")
    axs[1, 1].stairs(lambda_res, t_grid, color="C2", label=r"$\lambda_n$")
    axs[1, 1].set_title("contact force and distance")
    ax_dist = axs[1, 1].twinx()  # own y axis, force and distance have different scales
    ax_dist.plot(t_grid, np.linalg.norm(x_res[:, 0:2] - x_res[:, 2:4], axis=1) - 2*R,
                 color="C4", label="distance")
    ax_dist.set_ylabel("distance")
    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper left")
    ax_dist.legend(loc="upper right")
    for ax in axs[1]:
        ax.set_xlabel("$t$")
    fig.tight_layout()
    return fig


def _circle(center, r=R):
    tt = np.linspace(0, 2*np.pi, 100)
    return center[0] + r*np.cos(tt), center[1] + r*np.sin(tt)


def animate(solver):
    """Animate both balls with their traced paths and the target outline of ball 2."""
    x_res = solver.get("x")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid()
    ax.plot(*_circle(P2_REF), color="C3", alpha=0.4)

    (ball1,) = ax.plot([], [], "-", color="C0", lw=2, label="ball 1")
    (ball2,) = ax.plot([], [], "-", color="C3", lw=2, label="ball 2")
    (trail1,) = ax.plot([], [], "-", color="C0", alpha=0.3)
    (trail2,) = ax.plot([], [], "-", color="C3", alpha=0.3)
    ax.legend(loc="upper right")

    def update(k):
        ball1.set_data(*_circle(x_res[k, 0:2]))
        ball2.set_data(*_circle(x_res[k, 2:4]))
        trail1.set_data(x_res[:k+1, 0], x_res[:k+1, 1])
        trail2.set_data(x_res[:k+1, 2], x_res[:k+1, 3])
        return ball1, ball2, trail1, trail2

    


def example(plot=True):
    solver = solve_ocp()
    if plot:
        plot_results(solver)
        anim = animate(solver)
        plt.show()
    return solver


if __name__ == "__main__":
    example()
