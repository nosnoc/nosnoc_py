"""
Plotting helpers for the tethered-drone OCP result.

Three figures, each labeled and self-contained:
  1. XZ path: the flown trajectory vs. the reference, the ground line and the cable's reach
     (dashed circle of radius `cable_length` around the cable origin).
  2. Rotor thrusts (f_left, f_right) over time, with the [0, max_thrust] bounds marked.
  3. Contact diagnostics: gap functions (p_z for the ground, cable_length - dist for the cable)
     and the corresponding contact forces (lambda_normal), which should only be nonzero exactly
     where the matching gap is (numerically) zero -- the complementarity condition
     0 <= lambda_normal _|_ f_c(q) >= 0 made visible.
"""
import numpy as np
import matplotlib.pyplot as plt

from drone_cable_model import DroneCableConfig


def _lambda_time_grid(solver):
    """
    Time of every RK stage point, in the same flattened (stage, finite-element, RK-stage) order
    as `solver.get_full('lambda_normal')` / `solver.get_full('y_gap')`. `lambda_normal` has no
    left/right boundary point (see `discrete_time_problem/cls.py::_create_variables`), so it needs
    its own time grid rather than `get_time_grid_full()` (which includes those boundary points).
    """
    opts = solver.opts
    h = solver._fe_lengths()
    c = solver.dtp.rk.colloc_points()
    t, t_start = [], 0.0
    for jj in range(len(h)):
        for kk in range(opts.n_s):
            t.append(t_start + c[kk] * h[jj])
        t_start += h[jj]
    return np.array(t)


def plot_xz_path(solver, x_ref_full, cfg: DroneCableConfig, save_path=None):
    x_res = solver.get("x")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_ref_full[:, 0], x_ref_full[:, 1], "--", color="tab:purple", label="Reference path")
    ax.plot(x_res[:, 0], x_res[:, 1], "-", color="tab:blue", linewidth=2, label="Flown path")
    ax.plot(x_res[0, 0], x_res[0, 1], "o", color="black", zorder=5, label="Start")

    ax.axhline(0.0, color="saddlebrown", linewidth=2, label="Ground")

    # Cable reach (the drone's position is constrained to this disk).
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = cfg.cable_origin
    ax.plot(cx + cfg.cable_length * np.cos(theta), cy + cfg.cable_length * np.sin(theta),
             ":", color="gray", label="Cable reach")
    ax.plot(cx, cy, "s", color="gray", label="Cable anchor")

    x_span = np.concatenate([x_ref_full[:, 0], x_res[:, 0]])
    z_span = np.concatenate([x_ref_full[:, 1], x_res[:, 1]])
    pad = 0.3
    ax.set_xlim(x_span.min() - pad, x_span.max() + pad)
    ax.set_ylim(-0.2, max(z_span.max(), cy + cfg.cable_length) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_z$ [m]")
    ax.set_title("Tethered-drone flight path: climb, ellipse loop, descend")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig


def plot_thrusts(solver, cfg: DroneCableConfig, max_thrust: float, save_path=None):
    u_res = solver.get("u")
    t_control = solver.get_control_grid()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(t_control[:-1], u_res[:, 0], where="post", label="$f_{\\mathrm{left}}$")
    ax.step(t_control[:-1], u_res[:, 1], where="post", label="$f_{\\mathrm{right}}$")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.axhline(max_thrust, color="r", linestyle=":", label="Thrust bound")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("Rotor thrust [N]")
    ax.set_title("Rotor thrust commands")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig


def plot_contacts(solver, cfg: DroneCableConfig, save_path=None):
    t_full = solver.get_time_grid_full()
    x_full = solver.get_full("x")

    ground_gap = x_full[:, 1]
    diff = x_full[:, 0:2] - cfg.cable_origin
    cable_gap = cfg.cable_length - np.hypot(diff[:, 0], diff[:, 1])

    t_lambda = _lambda_time_grid(solver)
    lambda_normal = solver.get_full("lambda_normal")  # columns: [ground, cable]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax = axes[0]
    ax.plot(t_full, ground_gap, color="tab:brown", label="Ground gap $p_z$")
    ax_r = ax.twinx()
    ax_r.plot(t_lambda, lambda_normal[:, 0], ".", color="tab:red", markersize=3,
              label=r"Ground force $\lambda_{\mathrm{ground}}$")
    ax.set_ylabel("Gap [m]", color="tab:brown")
    ax_r.set_ylabel("Contact force [N]", color="tab:red")
    ax.set_title("Ground contact: gap vs. contact force")
    ax.axhline(0.0, color="k", linewidth=0.5)
    lines, labels = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines + lines_r, labels + labels_r, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_full, cable_gap, color="tab:gray", label="Cable gap (cable_length - dist)")
    ax_r = ax.twinx()
    ax_r.plot(t_lambda, lambda_normal[:, 1], ".", color="tab:purple", markersize=3,
              label=r"Cable tension $\lambda_{\mathrm{cable}}$")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("Gap [m]", color="tab:gray")
    ax_r.set_ylabel("Contact force [N]", color="tab:purple")
    ax.set_title("Cable contact: gap vs. contact force")
    ax.axhline(0.0, color="k", linewidth=0.5)
    lines, labels = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines + lines_r, labels + labels_r, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig
