r"""
An ellipse squeezed between two balls and lifted against gravity, as a CLS optimal control problem.

Two balls act as fingers to the left and to the right of an ellipse. Gravity acts on all three
bodies and there is no ground, so the *only* thing that can hold the ellipse up is Coulomb friction
at the two ball-ellipse contacts: the fingers must squeeze hard enough that
$2\mu\lambda_{\mathrm{n}} \ge m_{\mathrm{e}}(g + \ddot{y})$. The optimal control problem starts with
the balls held `cfg.initial_gap` clear of the ellipse, so nothing supports it and it free-falls
until the fingers close on it in a plastic impact. From there the scene is driven to a target in
which everything has been lifted by `cfg.h_lift`, with a regularization on the control, so the
optimizer has to find the cheapest grip that still catches and lifts.

The point of the example is the control parametrization, which is switched by `cfg.impedance` while
everything else stays fixed:

===============  ==========================================  ==========================
mode             force on ball `i`                           meaning of `u_i`
===============  ==========================================  ==========================
direct           $f_i = u_i$                                 force [N]
impedance        $f_i = K(u_i - p_i) - D v_i$                desired position [m]
===============  ==========================================  ==========================

Impedance control is the statement that there is an actual position and a position we *want* to be
in, with `K` and `D` deciding how hard the finger insists on closing that gap. The control
regularization in the cost penalizes the applied force in *both* modes, so the two runs are directly
comparable. The bounds `lbu`/`ubu` are not, since their units differ per mode.

The discretization is a CLS with implicit Euler time stepping (`use_fesd = False`), which is the
second order counterpart of the catching-up / time stepping scheme that a first order (projected
dynamical system) version of this scene would use. Two consequences of that choice show up below:

* a nonzero coefficient of restitution is rejected, every impact is plastic, and
* `lambda_normal` is a contact *impulse* over the step rather than a force, so it is divided by the
  step length wherever it is reported, see `contact_forces`.

Run it either way::

    python examples/impedance_grasp/ellipse_squeeze.py
    python -m examples.impedance_grasp.ellipse_squeeze
"""
import warnings

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import nosnoc

# Running the file directly leaves it without a parent package, so the relative import that the
# tests rely on cannot resolve. Running it as a script does put this directory on sys.path, so the
# sibling module is importable by plain name instead. Asking `__package__` rather than catching the
# ImportError keeps Python from warning about the attempt.

from geometry import (GRAVITY, GraspConfig, ball_slice, circle_outline, contact_terms,
                          ellipse_outline)

# ------------------------------------------------------------------ problem data

# Tracking weights. The height and the orientation of the ellipse are what the task is about, the
# balls only need enough weight to keep them from wandering off.
Q_DIAG = np.array([10.0, 100.0, 20.0, 1.0, 1.0, 1.0, 1.0] + [1.0]*7)
R_REG = 1       # Weight on the applied contact force, in both control modes.
Q_TERMINAL_SCALE = 100.0

UBU_FORCE = 10       # Control bound in direct mode [N].
UBU_POSITION = 0.5    # Control bound in impedance mode [m].

# Horizon discretization. With use_fesd = False these fix the step size h = T/(N_STAGES*N_FE).
N_STAGES = 30
N_FE = 2


def applied_forces(q, v, u, cfg: GraspConfig):
    """
    Control force on each ball, for the mode selected by `cfg.impedance`.

    Returned as a list of two 2-vectors so that the cost can regularize exactly the same quantity
    that the dynamics receive.
    """
    forces = []
    for i in range(2):
        u_i = u[2*i:2*i + 2]
        if cfg.impedance:
            forces.append(cfg.k_impedance*(u_i - ball_slice(q, i)) - cfg.d_impedance*ball_slice(v, i))
        else:
            forces.append(u_i)
    return forces


def get_grasp_model(cfg=None):
    """Build the squeeze-and-lift scene as a `nosnoc.model.Cls`."""
    cfg = cfg or GraspConfig()
    q = ca.SX.sym("q", 7)
    v = ca.SX.sym("v", 7)
    u = ca.SX.sym("u", 4)
    x = ca.vertcat(q, v)

    f_c, J_normal, J_tangent = contact_terms(q, v, cfg)
    f_1, f_2 = applied_forces(q, v, u, cfg)

    weight_ball = ca.vertcat(0.0, -cfg.m_ball*GRAVITY)
    f_v = ca.vertcat(0.0, -cfg.m_ellipse*GRAVITY, 0.0, f_1 + weight_ball, f_2 + weight_ball)

    x_err = x - ca.DM(cfg.x_ref)
    tracking = x_err.T@ca.diag(ca.DM(Q_DIAG))@x_err
    f_q = tracking + R_REG*(ca.sumsqr(f_1) + ca.sumsqr(f_2))
    f_q_T = Q_TERMINAL_SCALE*tracking

    ubu = UBU_POSITION if cfg.impedance else UBU_FORCE
    # The tangent Jacobian spans three bodies, so its columns are neither unit vectors nor
    # orthonormal, and the model warns twice about an anisotropic friction cone. Both checks assume
    # a single body contact where the column is a plain direction; here the columns are power dual,
    # which is what makes sum(lambda_t) <= mu*lambda_n exact Coulomb friction. See
    # `geometry.contact_terms`. (The orthonormality warning concerns the conic model, which this
    # example does not use anyway, planar contacts being polyhedral.)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*D_tangent are not unit vectors.*")
        warnings.filterwarnings("ignore", message=".*J_tangent for contact .* not orthonormal.*")
        return nosnoc.model.Cls(
            x=x,
            u=u,
            x0=cfg.x0,
            u0=cfg.u0,
            M=np.diag(cfg.inertia_diag),
            f_v=f_v,
            f_c=f_c,
            e=0.0,          # required: implicit Euler time stepping has no impulse variables
            
            J_normal=J_normal,
            J_tangent=J_tangent,
            lbu=-ubu*np.ones(4), ubu=ubu*np.ones(4),
            f_q=f_q,
            f_q_T=f_q_T,
            name="ellipse_squeeze",
        )


def get_default_options(cfg=None, **kwargs):
    cfg = cfg or GraspConfig()
    default_args = {
        "N_stages": N_STAGES,
        "N_finite_elements": N_FE,
        # Implicit Euler. Passing n_s and rk_scheme explicitly only avoids the warning from
        # `discrete_time_problem.Cls`, which would set exactly these values anyway.
        "n_s": 1,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": False,
        # The contacts are planar, so the tangent space is one dimensional and the polyhedral cone
        # spanned by [t, -t] is exact. FrictionModel.CONIC is rejected for planar contacts.
        "friction_model": nosnoc.FrictionModel.POLYHEDRAL,
        "T": cfg.T,
        # The defaults are 1, which is a poor guess for a scene that starts out of contact.
        "initial_lambda_normal": 0.0,
        "initial_y_gap": 0.0,
        "initial_lambda_tangent": 0.0,
        "initial_gamma_d": 0.0,
        "initial_beta_d": 0.0,
        "initial_delta_d": 0.0,
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_solver_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.sigma_0 = 1e1
    solver_opts.homotopy_update_slope = 0.1
    solver_opts.N_homotopy = 60
    solver_opts.complementarity_tol = 1e-6
    solver_opts.opts_casadi_nlp["ipopt"]["max_iter"] = 3000
    for k, val in kwargs.items():
        setattr(solver_opts, k, val)
    return solver_opts


def solve(cfg=None, opts=None, solver_opts=None):
    """Solve the OCP and return `(solver, cfg, stats)`."""
    cfg = cfg or GraspConfig()
    model = get_grasp_model(cfg)
    if opts is None:
        opts = get_default_options(cfg)
    if solver_opts is None:
        solver_opts = get_default_solver_options()
    solver = nosnoc.OcpSolver(model, opts, solver_opts)
    stats = solver.solve()
    return solver, cfg, stats


# ------------------------------------------------------------------ result extraction

def step_length(solver):
    """Fixed step length of the implicit Euler discretization."""
    opts = solver.opts
    return opts.T/np.sum(opts.N_finite_elements)


def control_grid(solver):
    """
    Time grid of the control stages.

    `OcpSolver.get_control_grid` cannot be used here: it reads the FESD step variables `w.h`
    unconditionally, which do not exist for `use_fesd = False`, so it raises. With a fixed step the
    grid is simply uniform.
    """
    if solver.opts.use_fesd:
        return solver.get_control_grid()
    return np.linspace(0.0, solver.opts.T, solver.opts.N_stages + 1)


def contact_forces(solver):
    """
    Normal and tangential contact *forces* at every finite element, shape `(n_fe, 2)` each.

    Without FESD the multipliers are impulses over the step, so they are divided by the step length
    here; with FESD they are already forces. `get` cannot be used for these: it assumes a value at
    the initial point, which stage only variables do not have, so it raises. `get_full` reports one
    row per finite element.
    """
    h = 1.0 if solver.opts.use_fesd else step_length(solver)
    lambda_n = np.atleast_2d(np.array(solver.get_full("lambda_normal")))/h
    #lambda_t = np.array(solver.get_full("lambda_tangent"))/h
    # The polyhedral generators come in [t, -t] pairs per contact, so the signed tangential force
    # of contact i is the difference of its two multipliers.
    n_c = lambda_n.shape[1]
    #ambda_t = lambda_t.reshape(lambda_t.shape[0], n_c, 2)
    #return lambda_n, lambda_t[:, :, 0] - lambda_t[:, :, 1]
    return lambda_n


def gap_values(solver, cfg):
    """Gap functions evaluated along the solution, shape `(n_points, 2)`."""
    q = ca.SX.sym("q", 7)
    v = ca.SX.sym("v", 7)
    f_c, _, _ = contact_terms(q, v, cfg)
    f_c_fun = ca.Function("f_c", [q], [f_c])
    x = solver.get("x")
    return np.vstack([np.array(f_c_fun(x[k, 0:7])).ravel() for k in range(x.shape[0])])


# ------------------------------------------------------------------ plotting

def animate(solver, cfg, save_path=None, show=True):
    """
    Animate the squeeze and lift.

    In impedance mode the desired position of each finger is drawn as a marker joined to the ball by
    a thin line, the virtual spring. The length of that line *is* the grip force divided by `K`,
    which makes the whole mechanism of impedance control visible.
    """
    nosnoc.latexify_plot()
    # A CLS reports both boundary states of every finite element so that velocity jumps are visible.
    # The positions are continuous, so every second row is a duplicate.
    x = solver.get("x")[::2]
    u = np.atleast_2d(solver.get("u"))
    n_frames = x.shape[0]
    # One control stage covers N_FE finite elements, hence one u row per N_FE frames.
    fe_per_stage = solver.opts.N_finite_elements[0]

    pad = cfg.a + 2*cfg.r_ball + 0.2
    x_lim = (np.min(x[:, [0, 3, 5]]) - pad, np.max(x[:, [0, 3, 5]]) + pad)
    y_lim = (np.min(x[:, [1, 4, 6]]) - pad, np.max(x[:, [1, 4, 6]]) + pad)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid()
    ax.set_title("impedance control" if cfg.impedance else "direct force control")

    # Target height of the ellipse center, as static scenery.
    ax.axhline(cfg.x_ref[1], color="k", ls=":", lw=1, alpha=0.6)

    (ellipse,) = ax.plot([], [], "-", color="C0", lw=2, label="ellipse")
    (ball1,) = ax.plot([], [], "-", color="C3", lw=2, label="fingers")
    (ball2,) = ax.plot([], [], "-", color="C3", lw=2)
    (trail,) = ax.plot([], [], "-", color="C0", alpha=0.3)
    (springs,) = ax.plot([], [], "-", color="C2", lw=1, alpha=0.8,
                         label="desired position" if cfg.impedance else None)
    (targets,) = ax.plot([], [], "x", color="C2", ms=7)
    ax.legend(loc="upper right")

    def update(frame):
        ellipse.set_data(*ellipse_outline(x[frame, 0:2], x[frame, 2], cfg))
        ball1.set_data(*circle_outline(x[frame, 3:5], cfg.r_ball))
        ball2.set_data(*circle_outline(x[frame, 5:7], cfg.r_ball))
        trail.set_data(x[:frame + 1, 0], x[:frame + 1, 1])
        if cfg.impedance:
            # Frame 0 is the initial state; frame k is the end of finite element k, which belongs to
            # control stage (k-1)//fe_per_stage.
            u_k = u[min(max(frame - 1, 0)//fe_per_stage, u.shape[0] - 1)]
            # A single polyline with a nan separating the two springs.
            springs.set_data([x[frame, 3], u_k[0], np.nan, x[frame, 5], u_k[2]],
                             [x[frame, 4], u_k[1], np.nan, x[frame, 6], u_k[3]])
            targets.set_data([u_k[0], u_k[2]], [u_k[1], u_k[3]])
        return ellipse, ball1, ball2, trail, springs, targets

    h0 = step_length(solver)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000*h0, blit=True)

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=max(1, int(1/h0))))
        print(f"  saved animation to {save_path}")
    if show:
        plt.show()
    return anim


def plot_time_series(solver, cfg):
    """Ellipse pose, contact forces against the friction cone, and the control."""
    nosnoc.latexify_plot()
    x = solver.get("x")
    t = solver.get_time_grid()
    u = np.atleast_2d(solver.get("u"))
    t_u = control_grid(solver)
    lambda_n, lambda_t = contact_forces(solver)
    t_lambda = np.linspace(0.0, solver.opts.T, lambda_n.shape[0] + 1)[1:]

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(t, x[:, 1], label="$y_{\\mathrm{e}}$")
    axs[0].axhline(cfg.x_ref[1], color="k", ls=":", lw=1, label="target")
    axs[0].set_ylabel("height [m]")
    axs[1].plot(t, x[:, 2], label=r"$\theta$")
    axs[1].set_ylabel("angle [rad]")
    for i in range(2):
        axs[2].plot(t_lambda, lambda_n[:, i], f"C{i}-", label=f"$\\lambda_{{n,{i+1}}}$")
        axs[2].plot(t_lambda, np.abs(lambda_t[:, i]), f"C{i}--", label=f"$|\\lambda_{{t,{i+1}}}|$")
        axs[2].plot(t_lambda, cfg.mu*lambda_n[:, i], f"C{i}:", alpha=0.7,
                    label=f"$\\mu\\lambda_{{n,{i+1}}}$")
    axs[2].axhline(cfg.m_ellipse*GRAVITY, color="k", ls="-.", lw=1, label="$m_{\\mathrm{e}}g$")
    axs[2].set_ylabel("contact force [N]")
    axs[3].step(t_u, np.vstack([u, u[-1]]), where="post")
    axs[3].set_ylabel("$u$ [m]" if cfg.impedance else "$u$ [N]")
    axs[3].set_xlabel("$t$ [s]")
    axs[3].legend(["$u_{1,x}$", "$u_{1,y}$", "$u_{2,x}$", "$u_{2,y}$"], ncol=2)
    for ax in axs[:3]:
        ax.legend(ncol=2)
    for ax in axs:
        ax.grid()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ example

def example(impedance=False, do_plot=True, save_gif=False, cfg=None):
    cfg = cfg or GraspConfig(impedance=impedance)
    solver, cfg, _ = solve(cfg)

    x = solver.get("x")
    #lambda_n, lambda_t = contact_forces(solver)
    lambda_n = contact_forces(solver)
    gaps = gap_values(solver, cfg)

    mode = "impedance" if cfg.impedance else "direct force"
    print(f"ellipse squeeze and lift ({mode} control)")
    print(f"  ellipse height {x[0, 1]:.3f} -> {x[-1, 1]:.3f} m (target {cfg.x_ref[1]:.3f})")
    print(f"  terminal angle {x[-1, 2]:+.4f} rad")
    print(f"  peak normal force {lambda_n.max():.1f} N "
          f"(a static hold needs {cfg.m_ellipse*GRAVITY/(2*cfg.mu):.2f} N per contact)")
    print(f"  min gap {gaps.min():+.2e} (penetration if negative)")
    #print(f"  min friction cone slack {np.min(cfg.mu*lambda_n - np.abs(lambda_t)):+.2e} N")

    if do_plot:
        save_path = f"ellipse_squeeze_{'impedance' if cfg.impedance else 'force'}.gif" \
            if save_gif else None
        animate(solver, cfg, save_path=save_path)
        plot_time_series(solver, cfg)
    return solver, cfg


if __name__ == "__main__":
    example(impedance=True)
