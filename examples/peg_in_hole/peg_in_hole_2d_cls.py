from pathlib import Path

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

import nosnoc
from nosnoc.model import Polytope2D, GrowthDistancePair2D, stack_growth_distances
from nosnoc.model.growth_distance import rotation_2d

# Planar peg-in-hole following Dietz et al., ECC 2025.
T = 6.0
N_STAGES = 40
TAUS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
TAU = TAUS[0]
PREPRINT_TAU_1 = 2.5e-3
U_MAX = 1.2
CLEARANCE = 0.05
Q0 = np.array([2.0, 3.3, -np.pi/4])
Q_REF = np.array([2.0, -1.0, 0.0])
PREPRINT = False  # simultaneous tau/sigma homotopy and lambda*y = tau, cf. Dietz et al., Sec. V-D
CCOPT = False # requires the activated env_ccopt (libMad on DYLD_LIBRARY_PATH)

PEG = Polytope2D.box(1.0, 2.0)
OCTAGON = Polytope2D.regular(8, 1.0/np.cos(np.pi/8), angle_offset=-np.pi/8)  # apothem 1, vertical faces
FLOOR = Polytope2D.box(6.0, 1.0)
ENVIRONMENT = [
    (OCTAGON, np.array([2.0 - 0.5 - CLEARANCE - 1.0, -1.0])),
    (OCTAGON, np.array([2.0 + 0.5 + CLEARANCE + 1.0, -1.0])),
    (FLOOR, np.array([2.0, -2.5])),
]
ENV_NAMES = ["left octagon", "right octagon", "floor"]


def peg_in_hole_model(relaxed_lp=True, tau0=TAU):
    q = ca.SX.sym("q", 3)
    v = ca.SX.sym("v", 3)
    u = ca.SX.sym("u", 3)
    tau = ca.SX.sym("tau")

    gd = stack_growth_distances([
        GrowthDistancePair2D(PEG, q[:2], q[2], P_env, rho_env, 0.0, q, tau, name=f"gd{k}")
        for k, (P_env, rho_env) in enumerate(ENVIRONMENT)
    ], relaxed=relaxed_lp)
    x_ref = np.concatenate([Q_REF, np.zeros(3)])

    model = nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        u=u,
        x0=np.concatenate([Q0, np.zeros(3)]),
        f_v=u,
        e=0,
        f_c=gd.f_c,
        J_normal=gd.J_normal,
        M=np.diag([1.0, 1.0, (1.0**2 + 2.0**2)/12]),
        z=gd.z, z0=gd.initial_guess(Q0, tau0), lbz=gd.lbz, ubz=gd.ubz,
        g_z=gd.g_z, lbg_z=gd.lbg_z, ubg_z=gd.ubg_z,
        p_global=tau, p_global_val=tau0,
        g_path=ca.sumsqr(u), ubg_path=U_MAX**2,
        f_q=ca.sumsqr(u),
        f_q_T=1e4*ca.sumsqr(ca.vertcat(q, v) - x_ref),
    )
    return model, gd


def get_default_options(**kwargs) -> nosnoc.Options:
    default_args = {
        "N_stages": N_STAGES,
        "N_finite_elements": 1,
        "n_s": 1,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": False,
        "eps_cls": 0.0,
        "T": T,
    }
    return nosnoc.Options(**(default_args | kwargs))


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    if CCOPT:
        solver_opts = nosnoc.mpccsol.plugins.ccopt.CCOptOptions()
        solver_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        solver_opts.madnlp_opts["tol"] = 1e-6
        solver_opts.madnlp_opts["max_iter"] = 3000
    else:
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        # The METIS ordering of the bundled MUMPS crashes (malloc abort) on the FESD-J problem, use AMD instead.
        solver_opts.opts_casadi_nlp["ipopt"]["mumps_pivot_order"] = 0
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


def solve_tau_homotopy(solver, taus=TAUS, sigma_0_warm=1e-4):
    """Solve for decreasing growth distance smoothing tau, each solve warm started from the previous one."""
    stats = []
    for k, tau in enumerate(taus):
        if k > 0:
            solver.warmstart()
            if isinstance(solver.solver_opts, nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions):
                solver.solver_opts.sigma_0 = sigma_0_warm
        solver.set_param("p_global", (), tau)
        solver.solve()
        stats.append(dict(solver.dtp.solver.stats))
    return stats


def solve_preprint_homotopy(solver, tau_1=PREPRINT_TAU_1, sigma_1=1.25e-3, kappa_tau=0.5, kappa_sigma=0.5,
                            mu_init_1=1.0, kappa_mu=0.1, n_hom=5):
    """Reduce tau, sigma and the IPOPT mu_init together, one NLP solve per step (Dietz et al., Sec. V-D)."""
    opts = solver.solver_opts
    opts.N_homotopy = 1
    tau, sigma, mu_init = tau_1, sigma_1, mu_init_1
    stats = []
    for k in range(n_hom):
        if k > 0:
            solver.warmstart(duals=True)
            tau, sigma, mu_init = kappa_tau*tau, kappa_sigma*sigma, kappa_mu*mu_init
        opts.sigma_0 = opts.sigma_N = sigma
        opts.opts_casadi_nlp["ipopt"]["mu_init"] = mu_init
        if k > 0:
            # IPOPT options are fixed at creation, so rebuild the NLP solver for the new mu_init.
            solver.dtp.solver.nlp.create_solver(opts.opts_casadi_nlp, plugin=opts.solver)
        solver.set_param("p_global", (), tau)
        solver.solve()
        stats.append(dict(solver.dtp.solver.stats))
    return stats


def _fe_end_states(solver):
    """Time and pre impact state at the end of every finite element, x0 first."""
    t_x = solver.get_time_grid()
    is_end = np.append(t_x[1:] != t_x[:-1], True)
    return t_x[is_end], solver.get("x")[is_end]


def plot_results(solver, gd):
    """Peg pose, controls, contact forces and growth distances."""
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
    for k, label in enumerate([r"$\rho_x$", r"$\rho_y$", r"$\theta$"]):
        axs[0, 0].plot(t_x, x_res[:, k], color=f"C{k}", label=label)
        axs[0, 0].axhline(Q_REF[k], color=f"C{k}", ls="--", alpha=0.5)
    axs[0, 0].set_title("peg pose (dashed: target)")
    for k, label in enumerate(["$u_x$", "$u_y$", r"$u_\theta$"]):
        axs[0, 1].stairs(u_res[:, k], t_stages, label=label)
    axs[0, 1].set_title("controls")
    for k in range(lambda_res.shape[1]):
        axs[1, 0].plot(t_colloc, lambda_res[:, k], ".-", label=f"peg - {ENV_NAMES[k]}")
        axs[1, 1].plot(t_fe, dist_res[:, k], label=f"peg - {ENV_NAMES[k]}")
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


def _world_vertices(P, rho, theta):
    return P.vertices()@np.array(rotation_2d(theta)).T + rho


def animate(solver, gif_path=None):
    _, x_res = _fe_end_states(solver)

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
