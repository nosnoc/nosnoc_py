"""
Tethered-drone OCP example -- solve and report.

Run this file directly:

    python run_drone_cable_example.py

"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(local_path)
results_path = os.path.join(local_path, "results")
os.makedirs(results_path, exist_ok=True)

from drone_cable_model import DroneCableConfig, DroneCableOCPConfig
from drone_cable_ocp import solve_drone_cable_ocp
import drone_cable_plot as plots


def print_summary(solver, x_ref_full, cfg: DroneCableConfig, ocp_cfg: DroneCableOCPConfig):
    x_res = solver.get("x")
    u_res = solver.get("u")
    objective = float(np.asarray(solver.get_objective()).item())

    ground_gap = x_res[:, 1]
    diff = x_res[:, 0:2] - cfg.cable_origin
    cable_gap = cfg.cable_length - np.hypot(diff[:, 0], diff[:, 1])
    final_err = np.abs(x_res[-1, :3] - x_ref_full[-1, :])

    print("=" * 70)
    print("TETHERED-DRONE OCP -- RESULT SUMMARY")
    print("=" * 70)
    print(f"Horizon:            T = {ocp_cfg.N_stages * ocp_cfg.sampling_time:.2f} s "
          f"({ocp_cfg.N_stages} stages x {ocp_cfg.N_finite_elements} finite elements)")
    print(f"Objective value:    {objective:.4f}")
    print("-" * 70)
    print("Feasibility checks (both should hold to solver tolerance):")
    print(f"  min ground gap  (p_z >= 0):                    {ground_gap.min(): .3e} m")
    print(f"  min cable gap   (cable_length - dist >= 0):    {cable_gap.min(): .3e} m")
    print(f"  rotor thrust range within [0, {ocp_cfg.max_thrust:.1f}] N:        "
          f"[{u_res.min():.3f}, {u_res.max():.3f}] N")
    print("-" * 70)
    print("Reference tracking:")
    print(f"  final position/pitch error (p_x, p_z, pitch):  {final_err}")
    print(f"  final state (p_x, p_z, pitch, v_x, v_z, v_pitch):")
    print(f"    {x_res[-1]}")
    print(f"  reference terminal (p_x, p_z, pitch):")
    print(f"    {x_ref_full[-1]}")
    print("=" * 70)


def main():
    cfg = DroneCableConfig()
    ocp_cfg = DroneCableOCPConfig()

    solver, x_ref_full = solve_drone_cable_ocp(cfg, ocp_cfg)
    print_summary(solver, x_ref_full, cfg, ocp_cfg)

    plots.plot_xz_path(solver, x_ref_full, cfg, save_path=os.path.join(results_path, "xz_path.png"))
    plots.plot_thrusts(solver, cfg, ocp_cfg.max_thrust, save_path=os.path.join(results_path, "thrusts.png"))
    plots.plot_contacts(solver, cfg, save_path=os.path.join(results_path, "contacts.png"))

    plt.show()


if __name__ == "__main__":
    main()
