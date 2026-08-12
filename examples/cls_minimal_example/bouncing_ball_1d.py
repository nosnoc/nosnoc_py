"""
One dimensional bouncing ball, the minimal example for the FESD-J discretization of a
Complementarity Lagrangian System (CLS).

The ball is dropped from a height q0 and bounces off the ground at q = 0. The impact is governed by
Newton's restitution law with the coefficient of restitution e. For e = 0 the impact is inelastic
and the ball stays on the ground, for e = 1 it is perfectly elastic and the ball keeps bouncing back
to its initial height.

The analytic solution is used to verify the accuracy of the discretization.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 9.81

X0 = np.array([0.8, 0.0])
T_SIM = 3.0
N_SIM = 30
N_FE = 2

def get_bouncing_ball_model(e=0.0, x0=X0):
    """Build the 1d bouncing ball as a `nosnoc.model.Cls`."""
    q = ca.SX.sym("q")
    v = ca.SX.sym("v")
    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=x0,
        M=np.eye(1),
        f_v=ca.SX(-GRAVITY),
        f_c=q,           # gap function, the ball contacts the ground at q = 0
        e=e,
        mu=0.0,          # frictionless
        name="bouncing_ball_1d",
    )


def get_default_options(**kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": N_FE,
        "n_s": 3,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": True,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_STAGE,
        "no_initial_impacts": True,
        "step_equilibration": nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,
        # A zero initial guess for the contact quantities works best for this example.
        "initial_Lambda_normal": 0.0,
        "initial_lambda_normal": 0.0,
        "initial_Y_gap": 0.0,
        "initial_y_gap": 0.0,
        "T": 1.0, #note that this gets overwirtten by T_sim / N_sim, redundancy-> TODO: evaluate this
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_integrator_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()

    solver_opts.homotopy_update_slope = 0.2
    solver_opts.N_homotopy = 15
    solver_opts.complementarity_tol = 1e-8
    default_args = {
        "T_sim": T_SIM,
        "N_sim": N_SIM,
        "solver_opts": solver_opts,
        "print_level": 0,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def analytic_solution(e, x0=X0, t_sim=T_SIM, n_points=2000):
    """
    Analytic trajectory of the bouncing ball.

    Returns the time grid, the positions, the velocities and the magnitudes of the impulses
    Lambda_normal = |v(t_s^+) - v(t_s^-)| = (1+e)|v(t_s^-)| at every impact.
    """
    q0, v0 = x0
    t_grid, q_traj, v_traj, impulses = [], [], [], []

    t, q, v = 0.0, q0, v0
    while t < t_sim:
        # time until the ball hits the ground: q + v*dt - g/2*dt^2 = 0
        dt_impact = (v + np.sqrt(max(v**2 + 2*GRAVITY*q, 0.0)))/GRAVITY
        if dt_impact <= 1e-12 or not np.isfinite(dt_impact):
            break
        dt = min(dt_impact, t_sim - t)
        tt = np.linspace(0.0, dt, max(int(n_points*dt/t_sim), 2))
        t_grid.append(t + tt)
        q_traj.append(q + v*tt - 0.5*GRAVITY*tt**2)
        v_traj.append(v - GRAVITY*tt)
        t += dt
        if dt < dt_impact:
            break
        # apply the impact law
        v_pre = v - GRAVITY*dt_impact
        impulses.append((1 + e)*abs(v_pre))
        q, v = 0.0, -e*v_pre
        if e == 0.0:
            # the ball stays on the ground for the remaining time
            tt = np.linspace(0.0, t_sim - t, 100)
            t_grid.append(t + tt)
            q_traj.append(np.zeros_like(tt))
            v_traj.append(np.zeros_like(tt))
            break

    return (np.concatenate(t_grid), np.concatenate(q_traj), np.concatenate(v_traj),
            np.array(impulses))


def solve_bouncing_ball(e=0.0, opts=None, integrator_opts=None, x0=X0):
    model = get_bouncing_ball_model(e=e, x0=x0)
    if opts is None:
        opts = get_default_options()
    if integrator_opts is None:
        integrator_opts = get_default_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, t_grid_full, x_res_full = integrator.simulate(x0)
    return t_grid, x_res, integrator


def plot_results(e, t_grid, x_res, integrator):
    nosnoc.latexify_plot()
    t_a, q_a, v_a, Lambda_a = analytic_solution(e)
    Lambda_num = integrator.get("Lambda_normal")


    #import pdb; pdb.set_trace()
    plt.figure(figsize=(7, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_grid, x_res[:, 0], "-o", markersize=3, label="$q$ - numerical")
    plt.plot(t_a, q_a, "--", label="$q$ - analytic")
    plt.ylabel("$q$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_grid, x_res[:, 1], "-o", markersize=3, label="$v$ - numerical")
    plt.plot(t_a, v_a, "--", label="$v$ - analytic")
    plt.ylabel("$v$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    if Lambda_num is not None:
        plt.stem(np.arange(len(Lambda_num.flatten())), Lambda_num.flatten(),
                 label=r"$\Lambda_\mathrm{n}$ - numerical")
    for ii, Lam in enumerate(Lambda_a):
        plt.axhline(Lam, color="k", linestyle="--",
                    label=r"$\Lambda_\mathrm{n}$ - analytic" if ii == 0 else None)
    plt.xlabel("finite element")
    plt.ylabel(r"$\Lambda_\mathrm{n}$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def example(e=0.0, plot=True):
    t_grid, x_res, integrator = solve_bouncing_ball(e=e)

    t_a, q_a, v_a, Lambda_a = analytic_solution(e)
    print(f"coefficient of restitution e = {e}")
    print(f"  position error {abs(q_a[-1] - x_res[-1, 0]):.2e}")
    print(f"  velocity error {abs(v_a[-1] - x_res[-1, 1]):.2e}")
    if len(Lambda_a):
        print(f"  first analytic impulse {Lambda_a[0]:.6f}")

    if plot:
        plot_results(e, t_grid, x_res, integrator)
    return t_grid, x_res, integrator


if __name__ == "__main__":
    example(e=1.0)
