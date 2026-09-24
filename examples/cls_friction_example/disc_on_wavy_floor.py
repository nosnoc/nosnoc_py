"""
Disc rolling and sliding on a wavy floor, integrated with the implicit Euler time-stepping scheme.

The example is planar, where the conic and the polyhedral friction model describe the *same*
friction cone: `D_tangent = [J_tangent, -J_tangent]` spans the interval `|lambda_t| <= mu*lambda_n`
that the conic model writes as `||lambda_t||_2 <= mu*lambda_n`. Running both and comparing the
trajectories is therefore a correctness cross check of the two reformulations against each other,
not a study of the polyhedral approximation error -- that only appears in 3D, where an m-gon
approximates a circle.
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc
# FrictionModel is not re-exported by nosnoc/__init__.py yet
from nosnoc.nosnoc_types import FrictionModel

A, K, R = 0.1, 2*np.pi/2.0, 0.2           # floor amplitude, wave number, disc radius
M_D, G = 1.0, 9.81
J_D = 0.5*M_D*R**2                        # moment of inertia of a disc
MU = 0.4                                  # coefficient of friction

X0 = np.array([0.0, 0.5, 0.0, 2.0, 0.0, 0.0])   # (x, y, theta, vx, vy, omega), launched sideways

T_SIM = 1.0
N_SIM = 50                                # integrator steps
N_FE = 2                                  # finite elements per step


def get_model():
    """The disc as a `nosnoc.model.Cls`, carrying both tangent matrices."""
    q = ca.SX.sym("q", 3)                 # x, y, theta
    v = ca.SX.sym("v", 3)

    s = A*ca.sin(K*q[0])                  # floor profile
    s_x = ca.jacobian(s, q[0])            # slope
    # grad f_c = (-s', 1, 0)^T has norm sqrt(1+s'^2), so w normalizes the gap function and both
    # contact Jacobians consistently. The conic cone ||lambda_t|| <= mu*lambda_n assumes that
    # J_normal and J_tangent are scaled alike, otherwise mu is not the friction coefficient.
    w = 1/ca.sqrt(1 + s_x**2)

    f_c = w*(q[1] - s) - R
    J_normal = ca.vertcat(-w*s_x, w, 0)                  # unit contact normal
    J_tangent = ca.vertcat(w, w*s_x, R)                  # unit tangent, R is the rolling lever arm
    D_tangent = ca.horzcat(J_tangent, -J_tangent)        # the two generators of the planar cone

    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=X0,
        M=np.diag([M_D, M_D, J_D]),
        f_v=ca.vertcat(0.0, -M_D*G, 0.0),
        f_c=f_c,
        J_normal=J_normal,
        J_tangent=J_tangent,                             # conic: orthonormal tangent basis
        D_tangent=D_tangent,                             # polyhedral: cone generators
        e=0.0,                                           # plastic impacts, required without FESD
        mu=MU,
        name="disc_on_wavy_floor",
    )


def get_options(friction_model):
    """
    Implicit Euler time stepping: `use_fesd = False` forces `n_s = 1` and Radau IIA, and the
    contact multipliers become impulses over the fixed step (see `_h_rescale`).
    """
    return nosnoc.Options(
        N_stages=1,
        N_finite_elements=N_FE,
        n_s=1,
        rk_scheme=nosnoc.RKScheme.RADAU_IIA,
        use_fesd=False,
        friction_model=friction_model,
        T=T_SIM/N_SIM,                    # one control stage = one integrator step
    )


def get_integrator_options():
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.N_homotopy = 15
    solver_opts.complementarity_tol = 1e-8
    return nosnoc.FESDIntegratorOptions(
        T_sim=T_SIM, N_sim=N_SIM, solver_opts=solver_opts, print_level=1)


def simulate(friction_model):
    model = get_model()
    integrator = nosnoc.Integrator(model, get_options(friction_model), get_integrator_options())
    t_grid, x_res, _, _ = integrator.simulate(X0)
    return t_grid, np.array(x_res), integrator


def energy(x):
    """Kinetic plus potential energy; with e = 0 and mu > 0 it must be non increasing."""
    q, v = x[:3, :], x[3:, :]
    kinetic = 0.5*(M_D*v[0, :]**2 + M_D*v[1, :]**2 + J_D*v[2, :]**2)
    return kinetic + M_D*G*q[1, :]


def main():
    results = {}
    for name, fm in (("conic", FrictionModel.CONIC),
                     ("polyhedral", FrictionModel.POLYHEDRAL)):
        t, x, integrator = simulate(fm)
        results[name] = (t, x)
        print(f"{name:11s}: energy {energy(x)[0]:.4f} -> {energy(x)[-1]:.4f}")

    (t_c, x_c), (t_p, x_p) = results["conic"], results["polyhedral"]
    # In 2D the two models are equivalent, so this is a hard cross check, not an approximation error.
    print(f"max |x_conic - x_polyhedral| = {np.max(np.abs(x_c - x_p)):.3e}")

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for name, (t, x) in results.items():
        axes[0].plot(x[0, :], x[1, :], label=name)
        axes[1].plot(t, x[3, :], label=f"{name}: v_x")
        axes[2].plot(t, energy(x), label=name)
    xs = np.linspace(x_c[0, :].min() - R, x_c[0, :].max() + R, 400)
    axes[0].plot(xs, A*np.sin(K*xs), "k-", lw=1, label="floor")
    axes[0].set_ylabel("y"); axes[0].set_xlabel("x"); axes[0].axis("equal")
    axes[1].set_ylabel("v_x")
    axes[2].set_ylabel("energy"); axes[2].set_xlabel("t")
    for ax in axes:
        ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
