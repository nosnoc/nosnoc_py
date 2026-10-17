import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nosnoc
import push_ball_pds as pds

N_FE = 2
C_DAMP = 50
# False: the Pds controls (velocities of ball 1) are applied directly as forces on ball 1.
# True: they are converted to accelerations by finite differences, so that ball 1 reaches the
# Pds velocity at the end of every control interval when it moves freely.
ACCELERATION_CONTROLS = False


def damped_model():
    """Cls push ball with u as force on ball 1 and viscous damping on ball 2."""
    p1, p2, v1, v2 = (ca.SX.sym(n, 2) for n in ("p1", "p2", "v1", "v2"))
    u = ca.SX.sym("u", 2)
    return nosnoc.model.Cls(
        x=ca.vertcat(p1, p2, v1, v2),
        u=u,
        x0=np.concatenate([pds.X0, np.zeros(4)]), 
        f_v=ca.vertcat(u, -C_DAMP*v2),
        f_c=ca.sumsqr(p1 - p2) - (2*pds.R)**2,
        e=0.0,
    )


def velocity_to_acceleration(u_pds):
    """Finite differences a_k = (u_k - u_{k-1})/h of the Pds velocities, with u_{-1} = 0 (ball 1 at rest)."""
    return np.diff(u_pds, axis=0, prepend=np.zeros((1, u_pds.shape[1])))/(pds.T/pds.N_STAGES)


def simulate(u_cls):
    """Apply the controls u_cls (forces on ball 1) to the Cls model, one integrator step per control interval."""
    model = damped_model()
    opts = nosnoc.Options(N_finite_elements=N_FE, n_s=1, rk_scheme=nosnoc.RKScheme.RADAU_IIA,
                          T=pds.T/pds.N_STAGES)
    int_opts = nosnoc.FESDIntegratorOptions(
        N_sim=pds.N_STAGES, T_sim=pds.T,
        solver_opts=nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions(print_level=0))
    t, x, _, _ = nosnoc.Integrator(model, opts, int_opts).simulate(model.x0, u=u_cls)
    return t, x


def plot_paths(x_pds, x_cls):
    """Paths of both balls, Pds OCP solution (dashed) vs. the same controls on Cls physics (solid)."""
    fig, ax = plt.subplots()
    ax.plot(x_pds[:, 0], x_pds[:, 1], "C0--", label="ball 1, Pds")
    ax.plot(x_pds[:, 2], x_pds[:, 3], "C3--", label="ball 2, Pds")
    ax.plot(x_cls[:, 0], x_cls[:, 1], "C0", label="ball 1, Cls")
    ax.plot(x_cls[:, 2], x_cls[:, 3], "C3", label="ball 2, Cls")
    ax.plot(*pds._circle(pds.P2_REF), "k:", label="target")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid()
    return fig


def plot_controls(u_pds):
    """Pds controls over time, piecewise constant on every control interval."""
    t_grid = np.linspace(0, pds.T, pds.N_STAGES+1)
    fig, ax = plt.subplots()
    ax.stairs(u_pds[:, 0], t_grid, label="$u_x$")
    ax.stairs(u_pds[:, 1], t_grid, label="$u_y$")
    ax.set_title("Pds controls (velocity of ball 1)")
    ax.set_xlabel("$t$")
    ax.legend()
    ax.grid()
    return fig


def plot_speed_pds(x_pds):
    """Speed |v| of both balls in the Pds OCP solution.

    Pds has no velocity state, the velocity is the position difference over every control interval.
    """
    t_grid = np.linspace(0, pds.T, pds.N_STAGES+1)
    v_pds = np.diff(x_pds, axis=0)/(pds.T/pds.N_STAGES)
    fig, ax = plt.subplots()
    ax.stairs(np.linalg.norm(v_pds[:, 0:2], axis=1), t_grid, color="C0", label="ball 1")
    ax.stairs(np.linalg.norm(v_pds[:, 2:4], axis=1), t_grid, color="C3", label="ball 2")
    ax.set_title("speed, Pds OCP solution")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|v\|$")
    ax.legend()
    ax.grid()
    return fig


def plot_speed_cls(t_cls, x_cls):
    """Speed |v| of both balls in the Cls simulation, velocity jumps show as vertical steps."""
    fig, ax = plt.subplots()
    ax.plot(t_cls, np.linalg.norm(x_cls[:, 4:6], axis=1), color="C0", label="ball 1")
    ax.plot(t_cls, np.linalg.norm(x_cls[:, 6:8], axis=1), color="C3", label="ball 2")
    ax.set_title("speed, Cls simulation of the Pds controls")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|v\|$")
    ax.legend()
    ax.grid()
    return fig


def animate(x_cls):
    """Animate the Cls simulation, one frame per control interval."""
    # x holds x0 and the (start, end) state of every finite element.
    x_frames = x_cls[0::2*N_FE]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.grid()
    ax.plot(*pds._circle(pds.P2_REF), "k:", label="target")
    (ball1,) = ax.plot([], [], "C0", lw=2, label="ball 1")
    (ball2,) = ax.plot([], [], "C3", lw=2, label="ball 2")
    (trail1,) = ax.plot([], [], "C0", alpha=0.3)
    (trail2,) = ax.plot([], [], "C3", alpha=0.3)
    ax.legend(loc="upper left")
    # axis limits from the trajectory, the balls can leave the Pds workspace
    pos = np.vstack([x_frames[:, 0:2], x_frames[:, 2:4]])
    ax.set_xlim(pos[:, 0].min() - 2*pds.R, pos[:, 0].max() + 2*pds.R)
    ax.set_ylim(pos[:, 1].min() - 2*pds.R, pos[:, 1].max() + 2*pds.R)

    def update(k):
        ball1.set_data(*pds._circle(x_frames[k, 0:2]))
        ball2.set_data(*pds._circle(x_frames[k, 2:4]))
        trail1.set_data(x_frames[:k+1, 0], x_frames[:k+1, 1])
        trail2.set_data(x_frames[:k+1, 2], x_frames[:k+1, 3])
        return ball1, ball2, trail1, trail2

    return FuncAnimation(fig, update, frames=len(x_frames), interval=1000*pds.T/pds.N_STAGES, blit=True)


def example(plot=True, acceleration_controls=ACCELERATION_CONTROLS):
    solver = pds.solve_ocp()
    x_pds = solver.get("x")
    u_pds = np.reshape(solver.get("u"), (pds.N_STAGES, -1))  # (N_sim, n_u), as simulate expects

    u_cls = velocity_to_acceleration(u_pds) if acceleration_controls else u_pds
    t_cls, x_cls = simulate(u_cls)
    if plot:
        plot_paths(x_pds, x_cls)
        plot_controls(u_pds)
        plot_speed_pds(x_pds)
        plot_speed_cls(t_cls, x_cls)
        anim = animate(x_cls)
        plt.gca().set_title("Cls simulation of the Pds controls")
        anim_pds = pds.animate(solver)  # the first order Pds OCP solution, for comparison
        plt.gca().set_title("Pds OCP solution")
        plt.show()
    return t_cls, x_cls


if __name__ == "__main__":
    example()