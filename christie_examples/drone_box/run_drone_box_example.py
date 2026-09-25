"""
Drone-towing-a-box-through-a-gap OCP example -- solve and report.

Run this file directly:

    python run_drone_box_example.py

It builds the drone+box `nosnoc.model.Cls` (`drone_box_model.py`), solves the reference-tracking
OCP with `nosnoc.OcpSolver` (`drone_box_ocp.py`), prints a labeled summary of the result, and
saves four figures
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(local_path)
results_path = os.path.join(local_path, "results")
os.makedirs(results_path, exist_ok=True)

from drone_box_model import DroneBoxConfig, DroneBoxOCPConfig, ellipse_params_bottom, ellipse_params_top, _outside_ellipse_expr
from drone_box_ocp import solve_drone_box_ocp
import drone_box_plot as plots


def print_summary(solver, x_ref_full, cfg: DroneBoxConfig, ocp_cfg: DroneBoxOCPConfig):
    x_res = solver.get("x")
    u_res = solver.get("u")
    objective = float(np.asarray(solver.get_objective()).item())

    drone_ground_gap = x_res[:, 1]
    box_ground_gap = x_res[:, 4]
    diff = x_res[:, 0:2] - x_res[:, 3:5]
    cable_gap = cfg.cable_length - np.hypot(diff[:, 0], diff[:, 1])

    cx_b, cz_b, a_b, b_b = ellipse_params_bottom(cfg)
    cx_t, cz_t, a_t, b_t = ellipse_params_top(cfg)
    clearances = np.array([
        np.asarray(_outside_ellipse_expr(x_res[:, 0], x_res[:, 1], cx_b, cz_b, a_b, b_b)).min(),
        np.asarray(_outside_ellipse_expr(x_res[:, 3], x_res[:, 4], cx_b, cz_b, a_b, b_b)).min(),
        np.asarray(_outside_ellipse_expr(x_res[:, 0], x_res[:, 1], cx_t, cz_t, a_t, b_t)).min(),
        np.asarray(_outside_ellipse_expr(x_res[:, 3], x_res[:, 4], cx_t, cz_t, a_t, b_t)).min(),
    ])

    box_final_err = np.abs(x_res[-1, 3:5] - ocp_cfg.x_final[3:5])
    box_final_v_err = np.abs(x_res[-1, 8:10] - ocp_cfg.v_final[3:5])

    print("=" * 70)
    print("DRONE-TOWING-A-BOX-THROUGH-A-GAP OCP -- RESULT SUMMARY")
    print("=" * 70)
    print(f"Horizon:            T = {ocp_cfg.N_stages * ocp_cfg.sampling_time:.2f} s "
          f"({ocp_cfg.N_stages} stages x {ocp_cfg.N_finite_elements} finite element)")
    print(f"Objective value:    {objective:.4f}")
    print("-" * 70)
    print("Feasibility checks (all should hold to solver tolerance):")
    print(f"  min drone-ground gap (p_z >= 0):                {drone_ground_gap.min(): .3e} m")
    print(f"  min box-ground gap   (b_z >= 0):                {box_ground_gap.min(): .3e} m")
    print(f"  min cable gap (cable_length - dist >= 0):       {cable_gap.min(): .3e} m")
    print(f"  min obstacle clearance (>= 0, all 4 checks):    {clearances.min(): .3e}")
    print(f"  rotor thrust range within [0, {ocp_cfg.max_thrust:.1f}] N:        "
          f"[{u_res.min():.3f}, {u_res.max():.3f}] N")
    print("-" * 70)
    print("Box target tracking (drone's own final position is not targeted):")
    print(f"  final box position error (b_x, b_z):   {box_final_err}")
    print(f"  final box velocity error (v_bx, v_bz): {box_final_v_err}")
    print(f"  final state (p_x, p_z, pitch, b_x, b_z, v_x, v_z, v_pitch, v_bx, v_bz):")
    print(f"    {x_res[-1]}")
    print("=" * 70)


def main():
    cfg = DroneBoxConfig()
    ocp_cfg = DroneBoxOCPConfig()

    solver, x_ref_full = solve_drone_box_ocp(cfg, ocp_cfg)
    print_summary(solver, x_ref_full, cfg, ocp_cfg)

    plots.plot_xz_path(solver, x_ref_full, cfg, ocp_cfg, save_path=os.path.join(results_path, "xz_path.png"))
    plots.plot_thrusts(solver, ocp_cfg.max_thrust, save_path=os.path.join(results_path, "thrusts.png"))
    plots.plot_contacts(solver, cfg, save_path=os.path.join(results_path, "contacts.png"))
    plots.plot_obstacle_clearance(solver, cfg, save_path=os.path.join(results_path, "obstacle_clearance.png"))

    plt.show()


if __name__ == "__main__":
    main()
