"""
Planar bouncing ball with Coulomb friction, the minimal example for CLS friction.

A ball is dropped onto the ground with a horizontal velocity. The impact is inelastic (e = 0), so
the ball lands and then slides until friction has brought it to a stop. Two things happen at the
impact and both are captured by FESD-J:

* the normal impulse `Lambda_normal` kills the vertical velocity, and
* the tangential impulse `Lambda_tangent`, bounded by `mu*Lambda_normal`, takes a bite out of the
  horizontal velocity in a single instant.

Afterwards the ball slides under a constant friction force `mu*m*g` until it sticks.

The contact is planar, so the tangent space is one dimensional and the polyhedral friction cone
spanned by `D_tangent = [t, -t]` is *exact* rather than an approximation. That is why
`FrictionModel.POLYHEDRAL` is the right choice here: it gives the same answer as the exact cone
while keeping the friction subproblem an LCP instead of an NCP. `FrictionModel.CONIC` is rejected
for planar contacts for exactly this reason.

The analytic solution is used to verify the discretization.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 9.81

X0 = np.array([0.0, 1.0, 4.0, 0.0])  # (q_x, q_y, v_x, v_y)
MU = 0.3
T_SIM = 1.2
N_SIM = 40
N_FE = 3


def get_bouncing_ball_2d_model(mu=MU, x0=X0):
    """Build the planar bouncing ball with friction as a `nosnoc.model.Cls`."""
    q = ca.SX.sym("q", 2)
    v = ca.SX.sym("v", 2)
    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=x0,
        M=np.eye(2),
        f_v=ca.vertcat(0.0, -GRAVITY),
        f_c=q[1],                    # gap function, the ball touches the ground at q_y = 0
        e=0.0,                       # inelastic impact
        mu=mu,
        # Tangent basis: one unit direction per contact, since the contact is planar. D_tangent is
        # built from this automatically as [t, -t].
        J_tangent=ca.DM([[1.0], [0.0]]),
        name="bouncing_ball_2d",
    )


def get_default_options(**kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": N_FE,
        "n_s": 2,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": True,
        "friction_model": nosnoc.FrictionModel.POLYHEDRAL,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_STAGE,
        "no_initial_impacts": True,
        "step_equilibration": nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,
        # A zero initial guess for the contact quantities works best for this example.
        "initial_Lambda_normal": 0.0,
        "initial_lambda_normal": 0.0,
        "initial_Y_gap": 0.0,
        "initial_y_gap": 0.0,
        "T": 1.0,  # overwritten by T_sim/N_sim, see bouncing_ball_1d.py
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
        "impact_guess_init": 7.0,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def analytic_solution(mu=MU, x0=X0, t_sim=T_SIM, n_points=2000):
    """
    Analytic trajectory: free flight, one inelastic impact with a friction impulse, then sliding
    under constant friction until the ball sticks.
    """
    qx0, qy0, vx0, _ = x0
    t_fall = np.sqrt(2*qy0/GRAVITY)
    v_normal = GRAVITY*t_fall                       # |v_y| just before the impact
    Lambda_n = v_normal                             # e = 0 and unit mass
    vx_post = max(vx0 - mu*Lambda_n, 0.0)           # tangential impulse, capped by the cone
    t_stop = vx_post/(mu*GRAVITY) if mu > 0 else np.inf

    t = np.linspace(0.0, t_sim, n_points)
    # Elapsed time in each phase: the flight clock stops at the impact, the sliding clock starts
    # there and stops again once the ball sticks. With those two, the positions are a single
    # expression covering all three phases.
    t_f = np.minimum(t, t_fall)
    s = np.clip(t - t_fall, 0.0, t_stop)

    qx = qx0 + vx0*t_f + vx_post*s - 0.5*mu*GRAVITY*s**2
    qy = qy0 - 0.5*GRAVITY*t_f**2                   # exactly 0 after the impact, by def. of t_fall
    # The velocities really do jump at the impact, so they need the case distinction.
    vx = np.where(t < t_fall, vx0, vx_post - mu*GRAVITY*s)
    vy = np.where(t < t_fall, -GRAVITY*t_f, 0.0)
    return t, qx, qy, vx, vy, Lambda_n, mu*Lambda_n


def solve_bouncing_ball_2d(mu=MU, opts=None, integrator_opts=None, x0=X0):
    model = get_bouncing_ball_2d_model(mu=mu, x0=x0)
    if opts is None:
        opts = get_default_options()
    if integrator_opts is None:
        integrator_opts = get_default_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, _, _ = integrator.simulate(x0)
    return t_grid, x_res, integrator


def plot_results(mu, t_grid, x_res):
    nosnoc.latexify_plot()
    t_a, qx_a, qy_a, vx_a, vy_a, _, _ = analytic_solution(mu)

    plt.figure(figsize=(7, 8))
    for idx, (num, ana, label) in enumerate([
            (x_res[:, 1], qy_a, "$q_y$"),
            (x_res[:, 0], qx_a, "$q_x$"),
            (x_res[:, 2], vx_a, "$v_x$")]):
        plt.subplot(3, 1, idx+1)
        plt.plot(t_grid, num, "-o", markersize=3, label=f"{label} - numerical")
        plt.plot(t_a, ana, "--", label=f"{label} - analytic")
        plt.ylabel(label)
        plt.grid()
        plt.legend()
    plt.xlabel("$t$")
    plt.tight_layout()
    plt.show()


def example(mu=MU, plot=True):
    t_grid, x_res, integrator = solve_bouncing_ball_2d(mu=mu)
    t_a, qx_a, qy_a, vx_a, _, Lambda_n, Lambda_t = analytic_solution(mu)

    print(f"coefficient of friction mu = {mu}")
    print(f"  q_x error {abs(qx_a[-1] - x_res[-1, 0]):.2e}")
    print(f"  v_x error {abs(vx_a[-1] - x_res[-1, 2]):.2e}")
    print(f"  analytic impulses: Lambda_n = {Lambda_n:.4f}, |Lambda_t| = {Lambda_t:.4f}")

    if plot:
        plot_results(mu, t_grid, x_res)
    return t_grid, x_res, integrator


if __name__ == "__main__":
    example()
