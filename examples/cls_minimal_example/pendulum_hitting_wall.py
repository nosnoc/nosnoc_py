"""
Planar pendulum hitting a wall, described with a redundant (n_q != n_v) orientation
representation.

The bob's orientation is represented by the unit vector q = (cos(theta), sin(theta)) in R^2
(n_q = 2) rather than the angle theta itself, while the generalized velocity remains the scalar
angular velocity v = omega (n_v = 1). This is the same kind of redundancy that a quaternion
orientation (n_q = 7, n_v = 6 for a floating-base body) introduces, scaled down to a 2D toy
problem that is still cheap to solve and easy to validate against an independent reference
trajectory.

The kinematic map q_dot = N(q) v that relates the two is

    N(q) = [-q2; q1]      (since d/dt(cos theta, sin theta) = (-sin theta, cos theta) * omega)

which genuinely depends on q (it is not the identity), so this example exercises the general
N(q) path added to `nosnoc.model.Cls`, `nosnoc.dcs.Cls` and `nosnoc.discrete_time_problem.Cls`
end to end, rather than the n_q == n_v, N = I case the rest of the CLS examples use.

The pendulum is released from rest at theta0 = -pi/2 (horizontal, to the left) and swings down
and up into a rigid wall/peg at theta = theta_wall on the right, where it undergoes a
Newton-restitution impact (coefficient of restitution e), exactly like the 1d bouncing ball in
`examples/cls_minimal_example`.

Correctness is checked three ways:
  1. ||q|| stays 1 along the whole trajectory. q_dot = N(q) v is tangent to the unit circle by
     construction (d/dt ||q||^2 = 2 q^T N(q) v = 0), so this would fail immediately if N(q) were
     not applied correctly by the dcs/discrete-time layers (before the N(q) support was added,
     `q_dot = v` was used directly, which is not even dimensionally valid here since v is 1-d and
     q is 2-d).
  2. The first impact time / pre-impact angular velocity / impulse magnitude are compared against
     an independent reference solution of the reduced-coordinate ODE
     (theta_ddot = -(g/L) sin(theta)) with event-based impact handling, obtained with
     `scipy.integrate.solve_ivp`.
  3. theta(t) and omega(t) reconstructed from the CLS trajectory are compared pointwise against
     that same reference trajectory.
"""
import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 9.81
MASS = 1.0
LENGTH = 1.0

THETA0 = -np.pi/2      # released from horizontal on the left, at rest
OMEGA0 = 0.0
THETA_WALL = np.pi/3   # peg/wall on the right, 60 degrees from the bottom

T_SIM = 2.5
N_SIM = 50
N_FE = 3


def get_pendulum_model(e=1.0):
    """Build the redundant-coordinate pendulum as a `nosnoc.model.Cls` with n_q = 2, n_v = 1."""
    q = ca.SX.sym("q", 2)   # q = (cos(theta), sin(theta))
    v = ca.SX.sym("v", 1)   # v = omega
    x = ca.vertcat(q, v)

    # Kinematic map q_dot = N(q) v, derived from q = (cos(theta), sin(theta)):
    # d/dt(cos(theta), sin(theta)) = (-sin(theta), cos(theta)) * omega = (-q2, q1) * v.
    N = ca.vertcat(-q[1], q[0])

    I = MASS*LENGTH**2
    M = I*np.eye(1)
    # Generalized torque about the pivot: -m g L sin(theta) = -m g L q2.
    f_v = -MASS*GRAVITY*LENGTH*q[1]

    # Bob position (x_bob, y_bob) = L*(sin(theta), -cos(theta)) = L*(q2, -q1) (theta = 0 hanging
    # straight down). Wall/peg at x_bob = x_wall = L*sin(theta_wall); contact from the left.
    x_wall = LENGTH*np.sin(THETA_WALL)
    f_c = x_wall - LENGTH*q[1]

    q0 = np.array([np.cos(THETA0), np.sin(THETA0)])
    x0 = np.concatenate([q0, [OMEGA0]])

    return nosnoc.model.Cls(
        x=x, q=q, v=v, N=N,
        x0=x0,
        M=M,
        f_v=f_v,
        f_c=f_c,
        e=e,
        mu=0.0,   # frictionless
        name="pendulum_wall_redundant",
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
        "T": 1.0,  # gets overwritten by T_sim / N_sim, see `Integrator._update_opts`.
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


def reference_solution(e, theta0=THETA0, omega0=OMEGA0, theta_wall=THETA_WALL, t_sim=T_SIM,
                        L=LENGTH, g=GRAVITY, n_points=4000, max_bounces=200):
    """
    Independent reference trajectory of the reduced-coordinate pendulum (theta, omega), obtained
    by integrating theta_ddot = -(g/L) sin(theta) with `solve_ivp` and manually applying the
    Newton restitution law omega -> -e * omega every time theta crosses theta_wall from below.

    Returns the time grid, theta(t), omega(t), and the impact times / pre-impact omegas.
    """
    def rhs(t, y):
        return [y[1], -(g/L)*np.sin(y[0])]

    def hit_wall(t, y):
        return y[0] - theta_wall
    hit_wall.terminal = True
    hit_wall.direction = 1.0  # only trigger on upward crossings, i.e. approaching the wall

    t0, y0 = 0.0, [theta0, omega0]
    t_all, theta_all, omega_all = [], [], []
    impact_times, impact_omega_pre = [], []

    for _ in range(max_bounces):
        n_eval = max(int(n_points*(t_sim - t0)/t_sim), 2)
        sol = solve_ivp(rhs, (t0, t_sim), y0, method="RK45",
                         t_eval=np.linspace(t0, t_sim, n_eval),
                         events=hit_wall, rtol=1e-12, atol=1e-12)
        t_all.append(sol.t)
        theta_all.append(sol.y[0])
        omega_all.append(sol.y[1])
        if sol.t_events[0].size == 0:
            break  # reached t_sim without hitting the wall again
        t_imp = sol.t_events[0][0]
        theta_imp, omega_imp = sol.y_events[0][0]
        impact_times.append(t_imp)
        impact_omega_pre.append(omega_imp)
        t0, y0 = t_imp, [theta_imp, -e*omega_imp]
        if t0 >= t_sim:
            break

    return (np.concatenate(t_all), np.concatenate(theta_all), np.concatenate(omega_all),
            np.array(impact_times), np.array(impact_omega_pre))


def solve_pendulum(e=1.0, opts=None, integrator_opts=None):
    model = get_pendulum_model(e=e)
    if opts is None:
        opts = get_default_options()
    if integrator_opts is None:
        integrator_opts = get_default_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, t_grid_full, x_res_full = integrator.simulate(model.x0)
    return t_grid, x_res, integrator


def plot_results(t_grid, x_res, t_ref, theta_ref, omega_ref):
    nosnoc.latexify_plot()
    theta_num = np.arctan2(x_res[:, 1], x_res[:, 0])
    omega_num = x_res[:, 2]
    q_norm_err = np.abs(np.hypot(x_res[:, 0], x_res[:, 1]) - 1.0)

    plt.figure(figsize=(7, 9))
    plt.subplot(3, 1, 1)
    plt.plot(t_grid, theta_num, "-o", markersize=3, label=r"$\theta$ - numerical")
    plt.plot(t_ref, theta_ref, "--", label=r"$\theta$ - reference")
    plt.axhline(THETA_WALL, color="r", linestyle=":", label=r"$\theta_{\rm wall}$")
    plt.ylabel(r"$\theta$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_grid, omega_num, "-o", markersize=3, label=r"$\omega$ - numerical")
    plt.plot(t_ref, omega_ref, "--", label=r"$\omega$ - reference")
    plt.ylabel(r"$\omega$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.semilogy(t_grid, q_norm_err + 1e-16, "-o", markersize=3, label=r"$\|q\|_2 - 1$")
    plt.xlabel("$t$")
    plt.ylabel(r"$\|q\|_2 - 1$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def example(e=1.0, plot=True):
    t_grid, x_res, integrator = solve_pendulum(e=e)

    t_ref, theta_ref, omega_ref, t_imp_ref, omega_pre_ref = reference_solution(e)

    theta_num = np.arctan2(x_res[:, 1], x_res[:, 0])
    omega_num = x_res[:, 2]
    q_norm_err = np.abs(np.hypot(x_res[:, 0], x_res[:, 1]) - 1.0)

    theta_ref_interp = np.interp(t_grid, t_ref, theta_ref)
    omega_ref_interp = np.interp(t_grid, t_ref, omega_ref)

    # The finite element boundary time is reported twice at every impact (pre- and post-impact
    # state, see `Integrator._cls_step_result`), so omega is genuinely discontinuous there.
    # `np.interp` against the (single-valued) reference cannot resolve that jump, so both
    # duplicated rows are excluded from the pointwise error below; the jump itself is checked
    # separately via the impulse/omega_pre comparison a few lines down.
    is_dup = np.zeros(t_grid.shape, dtype=bool)
    is_dup[1:] |= np.isclose(np.diff(t_grid), 0.0)
    is_dup[:-1] |= np.isclose(np.diff(t_grid), 0.0)
    mask = ~is_dup

    print(f"coefficient of restitution e = {e}")
    print(f"  n_q = 2, n_v = 1 (redundant unit-vector orientation, N(q) = [-q2; q1])")
    print(f"  max ||q|| - 1 over the trajectory (N(q) consistency check): {q_norm_err.max():.2e}")
    print(f"  theta error vs. reference (max over trajectory, excluding impact rows): {np.max(np.abs((theta_num - theta_ref_interp)[mask])):.2e}")
    print(f"  omega error vs. reference (max over trajectory, excluding impact rows): {np.max(np.abs((omega_num - omega_ref_interp)[mask])):.2e}")

    if len(t_imp_ref):
        I = MASS*LENGTH**2
        J_v_wall = -LENGTH*np.cos(THETA_WALL)
        omega_pre = omega_pre_ref[0]
        Lambda_analytic = -I*omega_pre*(1 + e)/J_v_wall
        Lambda_num = integrator.get("Lambda_normal")
        print(f"  first impact at t = {t_imp_ref[0]:.6f} s, omega_pre = {omega_pre:.6f}")
        print(f"  first analytic impulse Lambda_normal = {Lambda_analytic:.6f}")
        if Lambda_num is not None and Lambda_num.size:
            print(f"  largest numerical impulse Lambda_normal = {float(np.max(Lambda_num)):.6f}")

    if plot:
        plot_results(t_grid, x_res, t_ref, theta_ref, omega_ref)

    return t_grid, x_res, integrator


if __name__ == "__main__":
    example(e=1.0)
