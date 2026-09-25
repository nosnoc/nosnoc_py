"""
Validate the drone_ground OCP result with nosnoc's built-in Integrator.

What this does, in plain terms:
  1. Solve the OCP as usual (`drone_cable_ocp.solve_drone_cable_ocp`) -- this gives an optimal
     control sequence `u_ocp`, one thrust pair per control interval.
  2. Take *only* that control sequence and the starting state, and hand them to `nosnoc.Integrator`,
     which re-simulates the drone forward in time, interval by interval, on its own finer grid
     (`M` sub-steps per interval instead of the OCP's own 2).
  3. Compare the two trajectories. If they agree, the OCP's claimed trajectory is a trustworthy
     description of what the drone will actually do; if they don't, the OCP's own (coarser)
     discretization was too coarse to trust as-is.

The simulation below uses `use_fesd=False`, instead of `nosnoc`'s more elaborate 
adaptive-step-size mode (FESD, the mode the OCP itself uses).

Run this file directly:
    python run_drone_cable_simulation.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

local_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(local_path)
results_path = os.path.join(local_path, "results")
os.makedirs(results_path, exist_ok=True)

import nosnoc

from drone_cable_model import DroneCableConfig, DroneCableOCPConfig, build_drone_cable_model
from drone_cable_ocp import solve_drone_cable_ocp

# Sub-steps per OCP control interval used for the finer re-simulation. The OCP itself used 2
# finite elements per interval; this should be at least that, ideally a good deal more.
M = 4


def simulate_ocp_result(cfg: DroneCableConfig, ocp_cfg: DroneCableOCPConfig, solver, M: int = M):
    """
    Re-simulate the OCP's own optimal control sequence with nosnoc's Integrator, at a finer,
    independent time grid, as a validation check.

    Returns `(t_sim, x_sim)`, the simulated time grid and state trajectory.
    """
    u_ocp = solver.get("u")
    x0 = np.concatenate([ocp_cfg.x_init, ocp_cfg.v_init])

    # Same physics as the OCP
    model_sim = build_drone_cable_model(cfg, ocp_cfg, x_ref_T_val=np.zeros(cfg.nx))

    opts_sim = nosnoc.Options(
        N_stages=1,            # the Integrator always advances one control interval at a time
        N_finite_elements=M,   # sub-steps per interval -- the "how fine" knob
        T=1.0,                 # placeholder; overwritten by integrator_opts below
        use_fesd=False,        # fixed step size
        no_initial_impacts=True,
        print_level=0,
    )
    solver_opts_sim = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts_sim.complementarity_tol = ocp_cfg.complementarity_tol
    solver_opts_sim.N_homotopy = ocp_cfg.N_homotopy

    integrator_opts = nosnoc.FESDIntegratorOptions(
        N_sim=ocp_cfg.N_stages,                                # one simulate-step per OCP interval
        T_sim=ocp_cfg.N_stages * ocp_cfg.sampling_time,        # same total horizon as the OCP
        solver_opts=solver_opts_sim,
        print_level=0,
    )
    integrator = nosnoc.Integrator(model_sim, opts_sim, integrator_opts)

    t_sim, x_sim, _, _ = integrator.simulate(x0, u=u_ocp)
    return t_sim, x_sim


def main():
    cfg = DroneCableConfig()
    ocp_cfg = DroneCableOCPConfig()

    solver, x_ref_full = solve_drone_cable_ocp(cfg, ocp_cfg)
    x_ocp = solver.get("x")
    t_ocp = solver.get_time_grid()

    t_sim, x_sim = simulate_ocp_result(cfg, ocp_cfg, solver, M=M)
    
    final_err = np.abs(x_ocp[-1] - x_sim[-1])

    print("=" * 70)
    print(f"OCP vs. finer re-simulation (M = {M} sub-steps per interval, fixed step)")
    print("=" * 70)
    print(f"OCP final state:          {x_ocp[-1]}")
    print(f"Simulated final state:    {x_sim[-1]}")
    print(f"final-state error:        {final_err}")
    print(f"max final-state error:    {final_err.max():.4e}")
    print("-" * 70)
    print("Feasibility of the SIMULATED trajectory (should hold just as well as the OCP's own):")
    print(f"  min ground gap (p_z >= 0):                   {x_sim[:, 1].min(): .3e} m")
    cable_gap_sim = cfg.cable_length - np.hypot(x_sim[:, 0], x_sim[:, 1])
    print(f"  min cable gap (cable_length - dist >= 0):    {cable_gap_sim.min(): .3e} m")
    print("=" * 70)

    # Plot both trajectories on their own (different) time grids -- no interpolation needed to
    # compare them visually.
    nosnoc.latexify_plot()
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t_ocp, x_ocp[:, 0], "-", label="OCP")
    axes[0].plot(t_sim, x_sim[:, 0], "--", label=f"Simulated (M={M})")
    axes[0].set_ylabel("$p_x$ [m]")
    axes[0].set_title("OCP result vs. finer re-simulation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_ocp, x_ocp[:, 1], "-", label="OCP")
    axes[1].plot(t_sim, x_sim[:, 1], "--", label=f"Simulated (M={M})")
    axes[1].set_ylabel("$p_z$ [m]")
    axes[1].set_xlabel("$t$ [s]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(results_path, "ocp_vs_simulation.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
