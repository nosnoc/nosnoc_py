from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import nosnoc

import peg_in_hole_impedance_pds as imp_pds
import peg_in_hole_impedance_cls as imp_cls
from peg_in_hole_2d_cls import T, N_STAGES, TAUS, Q0, Q_REF, PEG, ENVIRONMENT, get_default_options, _world_vertices

N_FE = 3


def solve_pds():
    solver, _ = imp_pds.solve_ocp()
    u = np.reshape(solver.get("u"), (N_STAGES, -1))
    return solver.get_time_grid(), solver.get("x"), u


def simulate_cls(u):
    model, _ = imp_cls.peg_in_hole_model()
    opts = get_default_options(N_stages=1, N_finite_elements=N_FE)
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.opts_casadi_nlp["ipopt"]["mumps_pivot_order"] = 0
    int_opts = nosnoc.FESDIntegratorOptions(N_sim=N_STAGES, T_sim=T, solver_opts=solver_opts, print_level=2)
    integrator = nosnoc.Integrator(model, opts, int_opts)
    integrator.plugin.set_param("p_global", (), TAUS[-1])
    t, x, _, _ = integrator.simulate(model.x0, u=u)
    return t, x


def plot_comparison(t_pds, x_pds, t_cls, x_cls, u):
    q_pds_on_cls = np.column_stack([np.interp(t_cls, t_pds, x_pds[:, k]) for k in range(3)])
    err = x_cls[:, :3] - q_pds_on_cls
    v_pds = np.diff(x_pds[:, :3], axis=0)/np.diff(t_pds)[:, None]

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for k, label in enumerate([r"\rho_x", r"\rho_y", r"\theta"]):
        axs[0, 0].plot(t_pds, x_pds[:, k], color=f"C{k}", ls="--", label=f"${label}$ PDS")
        axs[0, 0].plot(t_cls, x_cls[:, k], color=f"C{k}", label=f"${label}$ CLS")
        axs[0, 0].plot(t_cls, x_cls[:, 6+k], color=f"C{k}", ls=":", alpha=0.6)
        axs[0, 0].axhline(Q_REF[k], color=f"C{k}", alpha=0.3)
        axs[0, 1].stairs(v_pds[:, k], t_pds, color=f"C{k}", ls="--", label=f"$\\dot{{{label}}}$ PDS")
        axs[0, 1].plot(t_cls, x_cls[:, 3+k], color=f"C{k}", label=f"$\\dot{{{label}}}$ CLS")
        axs[1, 0].plot(t_cls, err[:, k], color=f"C{k}", label=f"$\\Delta {label}$")
        axs[1, 1].stairs(u[:, k], np.linspace(0, T, N_STAGES+1), color=f"C{k}", label=f"$u_{k}$")
    axs[1, 0].plot(t_cls, np.linalg.norm(err, axis=1), "k", label=r"$\|\Delta q_c\|$")
    axs[0, 0].set_title("compliant pose (dashed: PDS, solid: CLS, dotted: reference)")
    axs[0, 1].set_title("compliant velocity")
    axs[1, 0].set_title("CLS - PDS")
    axs[1, 1].set_title("PDS controls (reference velocity)")
    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper right", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("$t$")
    fig.tight_layout()
    return fig


def animate(t_pds, x_pds, t_cls, x_cls, gif_path=None):
    is_end = np.append(t_cls[1:] != t_cls[:-1], True)
    t_cls, x_cls = t_cls[is_end], x_cls[is_end]
    q_pds = np.column_stack([np.interp(t_cls, t_pds, x_pds[:, k]) for k in range(3)])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 5)
    ax.grid()
    for P, rho in ENVIRONMENT:
        ax.add_patch(Polygon(_world_vertices(P, rho, 0.0), color="gray"))
    ax.add_patch(Polygon(_world_vertices(PEG, Q_REF[:2], Q_REF[2]), fill=False, ls="--", color="C3"))
    peg_pds = ax.add_patch(Polygon(_world_vertices(PEG, Q0[:2], Q0[2]), fill=False, color="C1", lw=2, label="PDS"))
    peg_cls = ax.add_patch(Polygon(_world_vertices(PEG, Q0[:2], Q0[2]), color="C0", alpha=0.7, label="CLS"))
    ax.legend(loc="upper right")

    def update(k):
        peg_pds.set_xy(_world_vertices(PEG, q_pds[k, :2], q_pds[k, 2]))
        peg_cls.set_xy(_world_vertices(PEG, x_cls[k, :2], x_cls[k, 2]))
        return peg_pds, peg_cls

    anim = FuncAnimation(fig, update, frames=len(t_cls), interval=1000*T/(len(t_cls)-1), blit=True)
    if gif_path is not None:
        anim.save(gif_path, writer="pillow", fps=(len(t_cls)-1)/T)
    return anim


def example(plot=True):
    t_pds, x_pds, u = solve_pds()
    t_cls, x_cls = simulate_cls(u)
    print(f"final q_c PDS: {x_pds[-1, :3]}, error to Q_REF: {np.linalg.norm(x_pds[-1, :3] - Q_REF):.3e}")
    print(f"final q_c CLS: {x_cls[-1, :3]}, error to Q_REF: {np.linalg.norm(x_cls[-1, :3] - Q_REF):.3e}")
    print(f"final v_c CLS: {x_cls[-1, 3:6]}")
    if plot:
        fig = plot_comparison(t_pds, x_pds, t_cls, x_cls, u)
        fig.savefig(Path(__file__).with_name(Path(__file__).stem + "_results.png"), dpi=150)
        anim = animate(t_pds, x_pds, t_cls, x_cls, gif_path=Path(__file__).with_suffix(".gif"))
        plt.show()
    return t_pds, x_pds, t_cls, x_cls, u


if __name__ == "__main__":
    example()
