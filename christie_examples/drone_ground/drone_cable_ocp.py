"""
Build and solve the tethered-drone reference-tracking OCP with `nosnoc`.

Run this file directly to solve the OCP and print a summary of the result.
"""
import os
import sys

import numpy as np

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(local_path)

import nosnoc

from drone_cable_model import DroneCableConfig, DroneCableOCPConfig
from drone_cable_model import build_drone_cable_model, build_reference_trajectory


def get_nosnoc_options(ocp_cfg: DroneCableOCPConfig):
    return nosnoc.Options(
        T=ocp_cfg.N_stages * ocp_cfg.sampling_time,     # total time horizon (could give h instead)
        N_stages=ocp_cfg.N_stages,                      # Nb of control intervals
        N_finite_elements=ocp_cfg.N_finite_elements,    # substeps per control interval for contact resolution
        n_s=ocp_cfg.n_s,                                # Runge-Kutta stages within each finite element
        rk_scheme=nosnoc.RKScheme.RADAU_IIA,
        # FESD is the adaptive step-size machinery
        use_fesd=True,                                  # Finite elements length become decision variables 
        step_equilibration=nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,                       
        cls_discretization=nosnoc.ClsDiscretization.FESD_J,
        cross_comp_mode=nosnoc.CrossComplementarityMode.FE_STAGE,   # Should try FE_FE because lighter, but less precise
        # The drone starts exactly on the ground (p_z = 0), so the initial state already sits at
        # the ground contact boundary. There is no impact "before time zero" to resolve there.
        no_initial_impacts=True,
        initial_lambda_normal=0.0,
        initial_Lambda_normal=0.0,
        initial_y_gap=0.0,
        initial_Y_gap=0.0,
        print_level=ocp_cfg.print_level,
    )


def get_solver_options(ocp_cfg: DroneCableOCPConfig):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.homotopy_update_slope = ocp_cfg.homotopy_update_slope
    solver_opts.N_homotopy = ocp_cfg.N_homotopy
    solver_opts.complementarity_tol = ocp_cfg.complementarity_tol
    solver_opts.print_level = ocp_cfg.print_level
    return solver_opts


def solve_drone_cable_ocp(cfg: DroneCableConfig = None, ocp_cfg: DroneCableOCPConfig = None):
    """
    Build the tethered-drone model + OCP, solve it, and return
    `(solver, x_ref_full, t_grid_ref)` where `x_ref_full` is the (N+1, nx) reference trajectory
    used for the cost (so callers can plot the numerical result against it).
    """
    cfg = cfg or DroneCableConfig()
    ocp_cfg = ocp_cfg or DroneCableOCPConfig()

    # Reference trajectory: one row per grid point (N_stages + 1 rows). Stage k (1..N_stages) is
    # tracked against row k (row 0 is the fixed initial state), and the terminal cost tracks the
    # very last row.
    x_ref_full = build_reference_trajectory(
        ocp_cfg.N_stages, cfg.nx,
        ellipse_center_z=ocp_cfg.ellipse_center_z, ellipse_a=ocp_cfg.ellipse_a,
        ellipse_b=ocp_cfg.ellipse_b, climb_frac=ocp_cfg.climb_frac, land_frac=ocp_cfg.land_frac,
    )

    # returns a nosnoc.model.Cls
    model = build_drone_cable_model(cfg, ocp_cfg, x_ref_T_val=x_ref_full[-1, :])

    opts = get_nosnoc_options(ocp_cfg)
    solver_opts = get_solver_options(ocp_cfg)
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    # Feed in the actual per-stage reference (row k for stage k).
    solver.set_param("p_time_var", (range(1, opts.N_stages + 1),), x_ref_full[1:, :])

    stats = solver.solve()
    print(f"Solver converged: {stats['converged']}, "
          f"total wall time: {stats['wall_time_total']:.2f} s")

    return solver, x_ref_full


if __name__ == "__main__":
    # For the full run with a labeled summary and plots, use `run_drone_cable_example.py`
    # instead -- this is just a quick solve-only smoke test.
    solver, x_ref_full = solve_drone_cable_ocp()
    x_res = solver.get("x")
    print(f"Objective value: {float(np.asarray(solver.get_objective()).item()):.4f}")
    print(f"Final state (p_x, p_z, pitch, v_x, v_z, v_pitch): {x_res[-1]}")
    print(f"Reference terminal position: {x_ref_full[-1]}")
