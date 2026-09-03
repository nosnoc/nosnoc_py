"""
Spatial bouncing ball with Coulomb friction, comparing the two friction models.

A ball is dropped onto a plane with a horizontal velocity that is not aligned with either tangent
axis. The impact is inelastic, so the ball lands and slides until friction stops it. In 3D the
tangent space is a plane, and the two friction models genuinely differ:

* `FrictionModel.CONIC` imposes the exact Coulomb cone ||lambda_t||_2 <= mu*lambda_n. Friction is
  isotropic, so it opposes the sliding direction exactly and the direction of motion is preserved.
* `FrictionModel.POLYHEDRAL` replaces the cone by the convex hull of the generators in
  `D_tangent`. With the four generators built automatically from `J_tangent`, that hull is the
  L1 ball inscribed in the friction disc: correct along the tangent axes, but only
  1/sqrt(2) ~ 71% as strong along the diagonals. The ball is therefore under-braked and its
  velocity drifts towards the diagonal where friction is weakest.

Pass a `D_tangent` with more generators to shrink that gap; the maximum error of an n-gon
inscribed in the disc is 1 - cos(pi/n), i.e. 29% for 4 generators, 7.6% for 8 and 1.9% for 16.

The conic model is compared against the analytic solution, which the polyhedral model is not
expected to match for this deliberately diagonal initial velocity.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 10.0

X0 = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])  # (q, v), sliding along (2,1)
MU = 0.2
T_SIM = 1.0
N_SIM = 30
N_FE = 2


def get_bouncing_ball_3d_model(mu=MU, x0=X0, n_facets=None):
    """
    Build the spatial bouncing ball with friction as a `nosnoc.model.Cls`.

    `n_facets` optionally requests a finer polyhedral cone: the generators are then spread evenly
    over the tangent plane instead of using the default four built from `J_tangent`.
    """
    q = ca.SX.sym("q", 3)
    v = ca.SX.sym("v", 3)
    # Orthonormal basis of the tangent plane, two columns for the single contact.
    J_tangent = ca.DM([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    D_tangent = None
    if n_facets is not None:
        if n_facets % 2 != 0:
            raise ValueError("n_facets must be even so that the friction cone stays symmetric.")
        angles = 2*np.pi*np.arange(n_facets)/n_facets
        D_tangent = ca.horzcat(*[np.cos(a)*J_tangent[:, 0] + np.sin(a)*J_tangent[:, 1]
                                 for a in angles])

    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=x0,
        M=np.eye(3),
        f_v=ca.vertcat(0.0, 0.0, -GRAVITY),
        f_c=q[2],
        e=0.0,
        mu=mu,
        J_tangent=J_tangent,
        D_tangent=D_tangent,
        name="bouncing_ball_3d",
    )


def get_default_options(friction_model=nosnoc.FrictionModel.CONIC, **kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": N_FE,
        "n_s": 2,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": True,
        "friction_model": friction_model,
        "conic_model_switch_handling": nosnoc.ConicModelSwitchHandling.ABS,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_STAGE,
        "no_initial_impacts": True,
        "step_equilibration": nosnoc.StepEquilibrationMode.HEURISTIC_MEAN,
        "initial_Lambda_normal": 0.0,
        "initial_lambda_normal": 0.0,
        "initial_Y_gap": 0.0,
        "initial_y_gap": 0.0,
        "T": 1.0,
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_integrator_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.homotopy_update_slope = 0.2
    solver_opts.N_homotopy = 15
    solver_opts.complementarity_tol = 1e-7
    default_args = {
        "T_sim": T_SIM,
        "N_sim": N_SIM,
        "solver_opts": solver_opts,
        "print_level": 0,
        "impact_guess_init": 7.0,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def analytic_solution(mu=MU, x0=X0, t_sim=T_SIM):
    """
    Exact (conic) solution: free flight, one inelastic impact whose friction impulse is capped by
    mu*Lambda_n, then sliding along the unchanged direction under a constant friction force.
    """
    qz0 = x0[2]
    v_t0 = np.array(x0[3:5])
    speed0 = np.linalg.norm(v_t0)
    direction = v_t0/speed0

    t_fall = np.sqrt(2*qz0/GRAVITY)
    Lambda_n = GRAVITY*t_fall
    speed_post = max(speed0 - mu*Lambda_n, 0.0)
    t_slide = min(max(t_sim - t_fall, 0.0), speed_post/(mu*GRAVITY) if mu > 0 else np.inf)
    speed_end = speed_post - mu*GRAVITY*t_slide

    q_end = (np.array(x0[0:2]) + v_t0*t_fall
             + direction*(speed_post*t_slide - 0.5*mu*GRAVITY*t_slide**2))
    return q_end, direction*speed_end, Lambda_n, mu*Lambda_n


def analytic_speed(t, mu=MU, x0=X0):
    """
    Analytic tangential speed |v_t| as a function of time, for the exact (conic) friction law.

    Constant during the free flight, then an instantaneous drop by `mu*Lambda_n` at the impact,
    then a linear decay at `mu*g` until the ball sticks.
    """
    t = np.asarray(t, dtype=float)
    speed0 = np.linalg.norm(x0[3:5])
    t_fall = np.sqrt(2*x0[2]/GRAVITY)
    speed_post = max(speed0 - mu*GRAVITY*t_fall, 0.0)   # mu*Lambda_n with Lambda_n = g*t_fall
    sliding = np.maximum(speed_post - mu*GRAVITY*np.clip(t - t_fall, 0.0, None), 0.0)
    return np.where(t <= t_fall, speed0, sliding)


def solve_bouncing_ball_3d(friction_model=nosnoc.FrictionModel.CONIC, mu=MU, x0=X0,
                           n_facets=None, opts=None, integrator_opts=None):
    model = get_bouncing_ball_3d_model(mu=mu, x0=x0, n_facets=n_facets)
    if opts is None:
        opts = get_default_options(friction_model=friction_model)
    if integrator_opts is None:
        integrator_opts = get_default_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, _, _ = integrator.simulate(x0)
    return t_grid, x_res, integrator


def example(mu=MU, plot=True):
    q_a, v_a, Lambda_n, Lambda_t = analytic_solution(mu)
    print(f"coefficient of friction mu = {mu}")
    print(f"  analytic impulses: Lambda_n = {Lambda_n:.4f}, |Lambda_t| = {Lambda_t:.4f}")
    print(f"  analytic terminal: q_t = ({q_a[0]:.4f}, {q_a[1]:.4f}), "
          f"|v_t| = {np.linalg.norm(v_a):.4f}")

    results = {}
    for label, kwargs in [
            ("conic", dict(friction_model=nosnoc.FrictionModel.CONIC)),
            ("polyhedral (4 facets)", dict(friction_model=nosnoc.FrictionModel.POLYHEDRAL)),
            ("polyhedral (16 facets)",
             dict(friction_model=nosnoc.FrictionModel.POLYHEDRAL, n_facets=16))]:
        t_grid, x_res, _ = solve_bouncing_ball_3d(mu=mu, **kwargs)
        results[label] = (t_grid, x_res)
        speed = np.linalg.norm(x_res[-1, 3:5])
        print(f"  {label:24s} q_t = ({x_res[-1,0]:.4f}, {x_res[-1,1]:.4f}), "
              f"|v_t| = {speed:.4f}   (|v_t| error {abs(speed-np.linalg.norm(v_a)):.2e})")

    if plot:
        plot_results(results, q_a, mu)
    return results


def plot_results(results, q_a, mu=MU):
    """
    Ground track and tangential speed side by side.

    The two panels show the two distinct errors of a coarse polyhedral cone: the track shows the
    *direction* drifting towards the diagonal, where the inscribed polygon is weakest, and the speed
    shows the *magnitude* being under-braked, both at the impact and during the slide.
    """
    nosnoc.latexify_plot()
    _, (ax_q, ax_v) = plt.subplots(1, 2, figsize=(12, 5))

    for label, (t_grid, x_res) in results.items():
        ax_q.plot(x_res[:, 0], x_res[:, 1], "-o", markersize=3, label=label)
        ax_v.plot(t_grid, np.linalg.norm(x_res[:, 3:5], axis=1), "-o", markersize=3, label=label)

    ax_q.plot([X0[0], q_a[0]], [X0[1], q_a[1]], "k--", label="analytic direction")
    ax_q.set_xlabel("$q_x$")
    ax_q.set_ylabel("$q_y$")
    ax_q.set_title("ground track: the coarse cone drifts towards the diagonal")
    ax_q.axis("equal")
    ax_q.grid()
    ax_q.legend()

    # The time grid of a CLS repeats the finite element boundary times, so the velocity jump at the
    # impact shows up as a genuine vertical segment rather than being interpolated across.
    t_dense = np.linspace(0.0, T_SIM, 1000)
    ax_v.plot(t_dense, analytic_speed(t_dense, mu), "k--", label="analytic")
    ax_v.set_xlabel("$t$")
    ax_v.set_ylabel(r"$\|v_\mathrm{t}\|$")
    ax_v.set_title("tangential speed: the drop at impact is the friction impulse")
    ax_v.grid()
    ax_v.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example()
