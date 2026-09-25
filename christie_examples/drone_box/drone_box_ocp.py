"""
Build and solve the drone-towing-a-box-through-a-gap OCP with `nosnoc`.

Run this file directly to solve the OCP and print a summary of the result.
"""
import os
import sys

import numpy as np

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(local_path)

import nosnoc

from drone_box_model import (
    DroneBoxConfig, DroneBoxOCPConfig,
    build_drone_box_model, build_initial_guess_through_gap,
)


def get_nosnoc_options(ocp_cfg: DroneBoxOCPConfig):
    return nosnoc.Options(
        N_stages=ocp_cfg.N_stages,
        N_finite_elements=ocp_cfg.N_finite_elements,
        n_s=ocp_cfg.n_s,
        rk_scheme=nosnoc.RKScheme.RADAU_IIA,
        T=ocp_cfg.N_stages * ocp_cfg.sampling_time,
        use_fesd=True,
        cls_discretization=nosnoc.ClsDiscretization.FESD_J,
        cross_comp_mode=nosnoc.CrossComplementarityMode.FE_STAGE,
        step_equilibration=nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,
        g_path_at_stg=True,
        # Both bodies start exactly on the ground (p_z = b_z = 0), so the initial state already
        # sits at the ground contact boundary -- no impact "before time zero" to resolve there.
        no_initial_impacts=True,
        initial_lambda_normal=0.0,
        initial_Lambda_normal=0.0,
        initial_y_gap=0.0,
        initial_Y_gap=0.0,
        print_level=ocp_cfg.print_level,
    )


def get_solver_options(ocp_cfg: DroneBoxOCPConfig):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.homotopy_update_slope = ocp_cfg.homotopy_update_slope
    solver_opts.N_homotopy = ocp_cfg.N_homotopy
    solver_opts.complementarity_tol = ocp_cfg.complementarity_tol
    solver_opts.print_level = ocp_cfg.print_level
    return solver_opts


def _apply_initial_guess(solver, opts, x_ref_full: np.ndarray, cfg: DroneBoxConfig):
    """
    Warm-start every state decision variable with the hand-threaded through-the-gap guess.
    """
    full_state_guess = np.hstack([x_ref_full, np.zeros_like(x_ref_full)])
    for ii in range(1, opts.N_stages + 1):
        for kk in range(opts.n_s + 1):
            solver.set("x", (ii, 1, kk), init=full_state_guess[ii, :])


def solve_drone_box_ocp(cfg: DroneBoxConfig = None, ocp_cfg: DroneBoxOCPConfig = None):
    """
    Build the drone-and-box model + OCP, solve it, and return `(solver, x_ref_full)`
    """
    cfg = cfg or DroneBoxConfig()
    ocp_cfg = ocp_cfg or DroneBoxOCPConfig()

    # Built before the model since it is also used as the running-cost tracking reference
    x_ref_full = build_initial_guess_through_gap(cfg, ocp_cfg, ocp_cfg.N_stages)

    model = build_drone_box_model(cfg, ocp_cfg, x_ref_full)

    opts = get_nosnoc_options(ocp_cfg)
    solver_opts = get_solver_options(ocp_cfg)
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    # Feed in the actual per-stage reference (row k for stage k).
    solver.set_param("p_time_var", (range(1, opts.N_stages + 1),), x_ref_full[1:, :])
    _apply_initial_guess(solver, opts, x_ref_full, cfg)

    stats = solver.solve()
    print(f"Solver converged: {stats['converged']}, "
          f"total wall time: {stats['wall_time_total']:.2f} s")

    return solver, x_ref_full


if __name__ == "__main__":
    # For the full run with a labeled summary and plots, use `run_drone_box_example.py` instead
    # -- this is just a quick solve-only smoke test.
    solver, x_ref_full = solve_drone_box_ocp()
    x_res = solver.get("x")
    print(f"Objective value: {float(np.asarray(solver.get_objective()).item()):.4f}")
    print(f"Final state: {x_res[-1]}")
    print(f"Reference/target terminal position: {x_ref_full[-1]}")
