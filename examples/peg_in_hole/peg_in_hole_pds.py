from pathlib import Path

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import nosnoc
from nosnoc.model import GrowthDistancePair2D, stack_growth_distances

from peg_in_hole_2d_cls import solve_tau_homotopy, solve_preprint_homotopy, T, N_STAGES, TAU, PREPRINT_TAU_1, U_MAX, Q0, Q_REF, PEG, ENVIRONMENT, ENV_NAMES, _world_vertices


PREPRINT = False


def peg_in_hole_model(relaxed_lp=True, tau0=TAU):
    q = ca.SX.sym("q", 3)
    u = ca.SX.sym("u", 3)
    tau = ca.SX.sym("tau")

    gd = stack_growth_distances([
        GrowthDistancePair2D(PEG, q[:2], q[2], P_env, rho_env, 0.0, q, tau, name=f"gd{k}")
        for k, (P_env, rho_env) in enumerate(ENVIRONMENT)
    ], relaxed=relaxed_lp)

    model = nosnoc.model.Pds(
        x=q,
        u=u,
        x0=Q0,
        f=u,
        f_c=gd.f_c,
        J_n=gd.J_normal,
        z=gd.z, z0=gd.initial_guess(Q0, tau0), lbz=gd.lbz, ubz=gd.ubz,
        g_z=gd.g_z, lbg_z=gd.lbg_z, ubg_z=gd.ubg_z,
        p_global=tau, p_global_val=tau0,
        g_path=ca.sumsqr(u), ubg_path=U_MAX**2,
        f_q=ca.sumsqr(u),
        f_q_T=1e4*ca.sumsqr(q - Q_REF),
    )
    return model, gd


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

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    if PREPRINT:
        model, gd = peg_in_hole_model(relaxed_lp=False, tau0=PREPRINT_TAU_1)
    else:
        model, gd = peg_in_hole_model()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    if PREPRINT:
        solve_preprint_homotopy(solver)
    else:
        solve_tau_homotopy(solver)

    return solver, gd


def plot_results(solver, gd):
    """Peg pose, controls, contact forces and growth distances."""
    x_res = solver.get("x")
    u_res = np.reshape(solver.get("u"), (N_STAGES, -1))
    lambda_res = np.reshape(solver.get_full("lambda_normal"), (N_STAGES, -1))
    f_c_fun = ca.Function("f_c", [gd.z], [gd.f_c])
    dist_res = np.array([np.ravel(f_c_fun(z)) for z in solver.get("z")])
    t_grid = np.linspace(0, T, N_STAGES+1)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for k, label in enumerate([r"$\rho_x$", r"$\rho_y$", r"$\theta$"]):
        axs[0, 0].plot(t_grid, x_res[:, k], color=f"C{k}", label=label)
        axs[0, 0].axhline(Q_REF[k], color=f"C{k}", ls="--", alpha=0.5)
    axs[0, 0].set_title("peg pose (dashed: target)")
    for k, label in enumerate(["$u_x$", "$u_y$", r"$u_\theta$"]):
        axs[0, 1].stairs(u_res[:, k], t_grid, label=label)
    axs[0, 1].set_title("controls (velocities)")
    for k in range(lambda_res.shape[1]):
        axs[1, 0].stairs(lambda_res[:, k], t_grid, label=f"peg - {ENV_NAMES[k]}")
        axs[1, 1].plot(t_grid[-dist_res.shape[0]:], dist_res[:, k], label=f"peg - {ENV_NAMES[k]}")
    axs[1, 1].axhline(0.0, color="k", lw=0.8)
    axs[1, 0].set_title(r"contact forces $\lambda_n$")
    axs[1, 1].set_title(r"growth distances $2\alpha$")
    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper right")
    for ax in axs[1]:
        ax.set_xlabel("$t$")
    fig.tight_layout()

    return fig


def animate(solver, gif_path=None):
    x_res = solver.get("x")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 5)
    ax.grid()
    for P, rho in ENVIRONMENT:
        ax.add_patch(Polygon(_world_vertices(P, rho, 0.0), color="gray"))
    ax.add_patch(Polygon(_world_vertices(PEG, Q_REF[:2], Q_REF[2]), fill=False, ls="--", color="C3"))
    peg = ax.add_patch(Polygon(_world_vertices(PEG, Q0[:2], Q0[2]), color="C0", alpha=0.8))
    (trail,) = ax.plot([], [], "-", color="C0", alpha=0.3)

    def update(k):
        peg.set_xy(_world_vertices(PEG, x_res[k, :2], x_res[k, 2]))
        trail.set_data(x_res[:k+1, 0], x_res[:k+1, 1])
        return peg, trail

    anim = FuncAnimation(fig, update, frames=x_res.shape[0], interval=1000*T/N_STAGES, blit=True)
    if gif_path is not None:
        anim.save(gif_path, writer="pillow", fps=(x_res.shape[0]-1)/T)
    return anim


def example(plot=True):
    solver, gd = solve_ocp()
    if plot:
        plot_results(solver, gd)
        anim = animate(solver, gif_path=Path(__file__).with_suffix(".gif"))
        plt.show()
    return solver


if __name__ == "__main__":
    example()
