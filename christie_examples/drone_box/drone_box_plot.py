"""
Plotting helpers for the drone-towing-a-box-through-a-gap OCP result.

Four figures, each labeled and self-contained:
  1. XZ path: drone path, box path, both obstacles (drawn as the inscribed ellipses actually
     enforced, see `drone_box_model.py`), ground line, start/target markers, and the cable shown
     at a handful of sampled times so the tow connection is visible.
  2. Rotor thrusts (f_left, f_right) over time, with the [0, max_thrust] bounds marked.
  3. Contact diagnostics: the three gap functions (drone-ground, box-ground, cable) and their
     contact forces (lambda_normal) -- the complementarity condition
     0 <= lambda_normal _|_ f_c(q) >= 0 made visible, same idea as `drone_ground`'s contact plot.
  4. Obstacle clearance: the four `g_path` values (>= 0 required) over time, showing how close
     each body gets to each obstacle while threading the gap.
"""
import numpy as np
import matplotlib.pyplot as plt

from drone_box_model import DroneBoxConfig, ellipse_params_bottom, ellipse_params_top, _outside_ellipse_expr


def _lambda_time_grid(solver):
    opts = solver.opts
    h = solver._fe_lengths()
    c = solver.dtp.rk.colloc_points()
    t, t_start = [], 0.0
    for jj in range(len(h)):
        for kk in range(opts.n_s):
            t.append(t_start + c[kk] * h[jj])
        t_start += h[jj]
    return np.array(t)


def _draw_obstacles(ax, cfg: DroneBoxConfig):
    theta = np.linspace(0, 2 * np.pi, 200)
    for params, label in ((ellipse_params_bottom(cfg), "Bottom obstacle"),
                           (ellipse_params_top(cfg), "Top obstacle")):
        cx, cz, a, b = params
        ax.fill(cx + a * np.cos(theta), cz + b * np.sin(theta), color="lightcoral", alpha=0.5,
                 label=label)


def plot_xz_path(solver, x_ref_full, cfg: DroneBoxConfig, ocp_cfg, save_path=None):
    x_res = solver.get("x")

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_obstacles(ax, cfg)

    ax.plot(x_ref_full[:, 0], x_ref_full[:, 1], "--", color="tab:blue", alpha=0.5,
            label="Drone guess/reference")
    ax.plot(x_ref_full[:, 3], x_ref_full[:, 4], "--", color="tab:orange", alpha=0.5,
            label="Box guess/reference")
    ax.plot(x_res[:, 0], x_res[:, 1], "-", color="tab:blue", linewidth=2, label="Drone path")
    ax.plot(x_res[:, 3], x_res[:, 4], "-", color="tab:orange", linewidth=2, label="Box path")

    # Cable, drawn at a handful of sampled times so the tow connection is visible without
    # cluttering the plot.
    n_show = 12
    idx = np.linspace(0, x_res.shape[0] - 1, n_show).astype(int)
    for i, k in enumerate(idx):
        ax.plot([x_res[k, 0], x_res[k, 3]], [x_res[k, 1], x_res[k, 4]], "-", color="gray",
                linewidth=0.8, alpha=0.6, label="Cable" if i == 0 else None)

    ax.plot(x_res[0, 0], x_res[0, 1], "o", color="tab:blue", zorder=5, label="Drone start")
    ax.plot(x_res[0, 3], x_res[0, 4], "o", color="tab:orange", zorder=5, label="Box start")
    ax.plot(ocp_cfg.x_final[3], ocp_cfg.x_final[4], "*", color="tab:green", markersize=15,
            zorder=5, label="Box target")

    ax.axhline(0.0, color="saddlebrown", linewidth=2, label="Ground")

    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_z$ [m]")
    ax.set_title("Drone towing a box through a gap")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig


def plot_thrusts(solver, max_thrust: float, save_path=None):
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


def plot_contacts(solver, cfg: DroneBoxConfig, save_path=None):
    t_full = solver.get_time_grid_full()
    x_full = solver.get_full("x")

    ground_drone_gap = x_full[:, 1]
    ground_box_gap = x_full[:, 4]
    diff = x_full[:, 0:2] - x_full[:, 3:5]
    cable_gap = cfg.cable_length - np.hypot(diff[:, 0], diff[:, 1])

    t_lambda = _lambda_time_grid(solver)
    lambda_normal = solver.get_full("lambda_normal")  # columns: [ground_drone, ground_box, cable]

    panels = [
        ("Drone-ground contact", ground_drone_gap, lambda_normal[:, 0], "tab:brown", "tab:red"),
        ("Box-ground contact", ground_box_gap, lambda_normal[:, 1], "tab:olive", "tab:red"),
        ("Cable contact", cable_gap, lambda_normal[:, 2], "tab:gray", "tab:purple"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for ax, (title, gap, lam, gap_color, lam_color) in zip(axes, panels):
        ax.plot(t_full, gap, color=gap_color, label="Gap")
        ax_r = ax.twinx()
        ax_r.plot(t_lambda, lam, ".", color=lam_color, markersize=3, label="Contact force")
        ax.set_ylabel("Gap [m]", color=gap_color)
        ax_r.set_ylabel("Contact force [N]", color=lam_color)
        ax.set_title(f"{title}: gap vs. contact force")
        ax.axhline(0.0, color="k", linewidth=0.5)
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("$t$ [s]")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig


def plot_obstacle_clearance(solver, cfg: DroneBoxConfig, save_path=None):
    t_full = solver.get_time_grid_full()
    x_full = solver.get_full("x")

    cx_b, cz_b, a_b, b_b = ellipse_params_bottom(cfg)
    cx_t, cz_t, a_t, b_t = ellipse_params_top(cfg)

    clearances = {
        "Drone vs. bottom obstacle": _outside_ellipse_expr(x_full[:, 0], x_full[:, 1], cx_b, cz_b, a_b, b_b),
        "Box vs. bottom obstacle": _outside_ellipse_expr(x_full[:, 3], x_full[:, 4], cx_b, cz_b, a_b, b_b),
        "Drone vs. top obstacle": _outside_ellipse_expr(x_full[:, 0], x_full[:, 1], cx_t, cz_t, a_t, b_t),
        "Box vs. top obstacle": _outside_ellipse_expr(x_full[:, 3], x_full[:, 4], cx_t, cz_t, a_t, b_t),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, clearance in clearances.items():
        ax.plot(t_full, np.asarray(clearance).flatten(), label=label)
    ax.axhline(0.0, color="k", linewidth=1.0, label="Obstacle boundary (must stay $\\geq 0$)")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("Ellipse clearance (dimensionless)")
    ax.set_title("Obstacle-avoidance path constraints")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig
