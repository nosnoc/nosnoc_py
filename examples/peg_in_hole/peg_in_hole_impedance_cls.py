from pathlib import Path

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import nosnoc
from nosnoc.model import GrowthDistancePair2D, stack_growth_distances

from peg_in_hole_2d_cls import (solve_tau_homotopy, T, N_STAGES, TAU, U_MAX, Q0, Q_REF, PEG, ENVIRONMENT,
                            get_default_options, _world_vertices, _fe_end_states)

# Impedance law of Dietz et al. (preprint, Sec. V-A) in 2D, the reference x_r = (q_r, v_r) is a kinematic
# integrator q_r' = v_r = u and pulls the compliant peg x_c = (q_c, v_c).
MASS = 1.0
INERTIA = (1.0**2 + 2.0**2)/12
K_T = 50.0
K_R = 5.0
D_T = 2*np.sqrt(K_T*MASS)
D_R = 2*np.sqrt(K_R*INERTIA)
DAMPING = np.diag([D_T, D_T, D_R])


def impedance_spring(q_r, q_c):
    """Stiffness part of the impedance wrench, k_r sin(dtheta) is the planar version of the quaternion law."""
    return ca.vertcat(K_T*(q_r[:2] - q_c[:2]), K_R*ca.sin(q_r[2] - q_c[2]))


def peg_in_hole_model():
    q_c = ca.SX.sym("q_c", 3)
    v_c = ca.SX.sym("v_c", 3)
    q_r = ca.SX.sym("q_r", 3)
    u = ca.SX.sym("u", 3)
    tau = ca.SX.sym("tau")

    gd = stack_growth_distances([
        GrowthDistancePair2D(PEG, q_c[:2], q_c[2], P_env, rho_env, 0.0, q_c, tau, name=f"gd{k}")
        for k, (P_env, rho_env) in enumerate(ENVIRONMENT)
    ])

    model = nosnoc.model.Cls(
        x=ca.vertcat(q_c, v_c, q_r),
        u=u,
        x0=np.concatenate([Q0, np.zeros(3), Q0]),
        f_v=impedance_spring(q_r, q_c) + DAMPING@(u - v_c),
        f_w=u,
        e=0,
        f_c=gd.f_c,
        J_normal=gd.J_normal,
        M=np.diag([MASS, MASS, INERTIA]),
        z=gd.z, z0=gd.initial_guess(Q0, TAU), lbz=gd.lbz, ubz=gd.ubz,
        g_z=gd.g_z, lbg_z=gd.lbg_z, ubg_z=gd.ubg_z,
        p_global=tau, p_global_val=TAU,
        g_path=ca.sumsqr(u), ubg_path=U_MAX**2,
        f_q=ca.sumsqr(u),
        f_q_T=1e4*ca.sumsqr(ca.vertcat(q_c - Q_REF, v_c)) + 1e2*ca.sumsqr(q_r - Q_REF),
    )
    return model, gd


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    # The METIS ordering of the bundled MUMPS crashes (malloc abort) on the FESD-J problem, use AMD instead.
    solver_opts.opts_casadi_nlp["ipopt"]["mumps_pivot_order"] = 0
    model, gd = peg_in_hole_model()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    solve_tau_homotopy(solver)

    return solver, gd


def plot_results(solver, gd, q_r_idx=slice(6, 9)):
    """Compliant and reference pose, reference velocity, contact forces and growth distances."""
    t_x = solver.get_time_grid()
    x_res = solver.get("x")
    t_fe, _ = _fe_end_states(solver)
    t_stages = np.linspace(0, T, N_STAGES+1)
    t_colloc = (t_fe[:-1, None] + np.outer(np.diff(t_fe), solver.dtp.rk.colloc_points())).ravel()
    u_res = np.reshape(solver.get("u"), (N_STAGES, -1))
    lambda_res = np.array(solver.get_full("lambda_normal"))
    f_c_fun = ca.Function("f_c", [gd.z], [gd.f_c])
    dist_res = np.array([np.ravel(f_c_fun(z)) for z in solver.get("z")])

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for k, label in enumerate([r"\rho_x", r"\rho_y", r"\theta"]):
        axs[0, 0].plot(t_x, x_res[:, k], color=f"C{k}", label=f"${label}$")
        axs[0, 0].plot(t_x, x_res[:, q_r_idx][:, k], color=f"C{k}", ls=":", label=f"${label},r$")
        axs[0, 0].axhline(Q_REF[k], color=f"C{k}", ls="--", alpha=0.5)
    axs[0, 0].set_title("compliant pose (dotted: reference, dashed: target)")
    for k, label in enumerate([r"$v_{r,x}$", r"$v_{r,y}$", r"$\omega_r$"]):
        axs[0, 1].stairs(u_res[:, k], t_stages, label=label)
    axs[0, 1].set_title("controls (reference velocity)")
    for k in range(lambda_res.shape[1]):
        axs[1, 0].plot(t_colloc, lambda_res[:, k], ".-", label=fr"$\lambda_{{n,{k}}}$")
        axs[1, 1].plot(t_fe, dist_res[:, k], label=fr"$2\alpha_{k}$")
    axs[1, 0].set_title("contact forces")
    axs[1, 1].set_title("growth distances")
    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper right")
    for ax in axs[1]:
        ax.set_xlabel("$t$")
    fig.tight_layout()

    return fig


def animate(solver, gif_path=None, q_c_idx=slice(0, 3), q_r_idx=slice(6, 9)):
    _, x_res = _fe_end_states(solver)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 5)
    ax.grid()
    for P, rho in ENVIRONMENT:
        ax.add_patch(Polygon(_world_vertices(P, rho, 0.0), color="gray"))
    ax.add_patch(Polygon(_world_vertices(PEG, Q_REF[:2], Q_REF[2]), fill=False, ls="--", color="C3"))
    reference = ax.add_patch(Polygon(_world_vertices(PEG, Q0[:2], Q0[2]), fill=False, ls=":", color="C1"))
    peg = ax.add_patch(Polygon(_world_vertices(PEG, Q0[:2], Q0[2]), color="C0", alpha=0.8))
    (trail,) = ax.plot([], [], "-", color="C0", alpha=0.3)

    def update(k):
        q_c = x_res[k, q_c_idx]
        q_r = x_res[k, q_r_idx]
        peg.set_xy(_world_vertices(PEG, q_c[:2], q_c[2]))
        reference.set_xy(_world_vertices(PEG, q_r[:2], q_r[2]))
        trail.set_data(x_res[:k+1, q_c_idx][:, 0], x_res[:k+1, q_c_idx][:, 1])
        return peg, reference, trail

    anim = FuncAnimation(fig, update, frames=x_res.shape[0], interval=1000*T/(x_res.shape[0]-1), blit=True)
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
