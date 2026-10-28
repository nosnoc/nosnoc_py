from pathlib import Path

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc
from nosnoc.model import GrowthDistancePair2D, stack_growth_distances

from peg_in_hole_2d_cls import solve_tau_homotopy, TAU, U_MAX, Q0, Q_REF, PEG, ENVIRONMENT
from peg_in_hole_pds import get_default_options
from peg_in_hole_impedance_cls import DAMPING, impedance_spring, plot_results, animate

# Quasistatic limit m -> 0 of the impedance CLS: 0 = spring + D (v_r - q_c') + J_n lambda,
# so the contact multiplier is scaled by the damping, the physical contact force is D lambda.


def peg_in_hole_model():
    q_c = ca.SX.sym("q_c", 3)
    q_r = ca.SX.sym("q_r", 3)
    u = ca.SX.sym("u", 3)
    tau = ca.SX.sym("tau")

    gd = stack_growth_distances([
        GrowthDistancePair2D(PEG, q_c[:2], q_c[2], P_env, rho_env, 0.0, q_c, tau, name=f"gd{k}")
        for k, (P_env, rho_env) in enumerate(ENVIRONMENT)
    ])

    model = nosnoc.model.Pds(
        x=ca.vertcat(q_c, q_r),
        u=u,
        x0=np.concatenate([Q0, Q0]),
        f=ca.vertcat(u + np.linalg.inv(DAMPING)@impedance_spring(q_r, q_c), u),
        f_c=gd.f_c,
        J_n=ca.vertcat(gd.J_normal, ca.SX.zeros(3, gd.f_c.size1())),
        z=gd.z, z0=gd.initial_guess(Q0, TAU), lbz=gd.lbz, ubz=gd.ubz,
        g_z=gd.g_z, lbg_z=gd.lbg_z, ubg_z=gd.ubg_z,
        p_global=tau, p_global_val=TAU,
        g_path=ca.sumsqr(u), ubg_path=U_MAX**2,
        f_q=ca.sumsqr(u),
        f_q_T=1e4*ca.sumsqr(q_c - Q_REF) + 1e2*ca.sumsqr(q_r - Q_REF),
    )
    return model, gd


def solve_ocp(opts=None):
    if opts is None:
        opts = get_default_options()

    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    model, gd = peg_in_hole_model()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)

    solve_tau_homotopy(solver)

    return solver, gd


def example(plot=True):
    solver, gd = solve_ocp()
    if plot:
        plot_results(solver, gd, q_r_idx=slice(3, 6))
        anim = animate(solver, gif_path=Path(__file__).with_suffix(".gif"), q_r_idx=slice(3, 6))
        plt.show()
    return solver


if __name__ == "__main__":
    example()
