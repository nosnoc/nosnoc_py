import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nosnoc


A, K, R = 0.1, 2*np.pi/2.0, 0.2           # floor amplitude, wave number, disc radius
M_D, G = 1.0, 9.81
J_D = 0.5*M_D*R**2                        # moment of inertia of the disc
MU = 0.4                                

X0 = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0])  

T_SIM = 100                           
H_STEP = 0.01                           
N_FE = 2                                 
N_SIM = max(1, round(T_SIM/(H_STEP*N_FE))) 


def get_model():
    """The disc as a `nosnoc.model.Cls`, carrying both tangent matrices."""
    q = ca.SX.sym("q", 3)                 # x, y, theta of the disc
    v = ca.SX.sym("v", 3)

    s = A*ca.sin(K*q[0])                  # floor profile
    s_x = ca.jacobian(s, q[0])            # slope
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
        J_tangent=J_tangent,                             
        D_tangent=D_tangent,                         
        e=0.0,                                           
        mu=MU,
        name="disc_on_wavy_floor",
    )


def get_options():
    return nosnoc.Options(
        N_stages=1,
        N_finite_elements=N_FE,
        n_s=1,
        rk_scheme=nosnoc.RKScheme.RADAU_IIA,
        use_fesd=False,
        friction_model=nosnoc.FrictionModel.CONIC,
        T=T_SIM/N_SIM,                   
    )


def get_integrator_options():
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver_opts.N_homotopy = 15
    solver_opts.complementarity_tol = 1e-8
    return nosnoc.FESDIntegratorOptions(
        T_sim=T_SIM, N_sim=N_SIM, solver_opts=solver_opts, print_level=1)


def simulate():
    model = get_model()
    opts = get_options()
    integrator_opts = get_integrator_options()
    integrator = nosnoc.Integrator(model, opts, integrator_opts)
    t_grid, x_res, _, _ = integrator.simulate(X0)
    
    return t_grid, np.array(x_res).T, integrator


def energy(x):
    """Kinetic plus potential energy; with e = 0 and mu > 0 it must be non increasing."""
    q, v = x[:3, :], x[3:, :]
    kinetic = 0.5*(M_D*v[0, :]**2 + M_D*v[1, :]**2 + J_D*v[2, :]**2)
    return kinetic + M_D*G*q[1, :]


def contact_quantities(x):
    """Gap function and tangential contact velocity along the trajectory."""
    q, v = x[:3, :], x[3:, :]
    s = A*np.sin(K*q[0, :])
    s_x = A*K*np.cos(K*q[0, :])
    w = 1/np.sqrt(1 + s_x**2)
    f_c = w*(q[1, :] - s) - R
    # v_t = J_tangent^T v with J_tangent = (w, w*s_x, R)
    v_t = w*v[0, :] + w*s_x*v[1, :] + R*v[2, :]
    return f_c, v_t



def animate(t, x, save_path=None):
    """
    Animate the disc on the floor. The spoke is what makes the friction visible: while the disc
    slides it keeps pointing the same way, once it rolls it turns with the travelled distance.
    """
    floor_x = np.linspace(x[0, :].min() - 2*R, x[0, :].max() + 2*R, 600)
    floor_y = A*np.sin(K*floor_x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(floor_x, floor_y, "k-", lw=1.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(floor_x[0], floor_x[-1])
    ax.set_ylim(floor_y.min() - 0.1, x[1, :].max() + 2*R)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    disc = plt.Circle((x[0, 0], x[1, 0]), R, fill=False, lw=2, color="C0")
    ax.add_patch(disc)
    spoke, = ax.plot([], [], color="C3", lw=2)
    trace, = ax.plot([], [], color="C0", lw=0.8, alpha=0.4)
    label = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    _, v_t = contact_quantities(x)

    def update(k):
        cx, cy, th = x[0, k], x[1, k], x[2, k]
        disc.center = (cx, cy)
        spoke.set_data([cx, cx + R*np.sin(th)], [cy, cy - R*np.cos(th)])
        trace.set_data(x[0, :k+1], x[1, :k+1])
        label.set_text(f"t = {t[k]:5.3f} s    v_tangential = {v_t[k]:+6.3f}"
                       f"    {'rollt' if abs(v_t[k]) < 1e-3 else 'gleitet'}")
        return disc, spoke, trace, label

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000*T_SIM/len(t),
                         blit=False, repeat=False)
    if save_path is not None:
        anim.save(save_path, writer="pillow", fps=30)
        print(f"animation written to {save_path}")
    return anim

def main():
    t, x, integrator = simulate()
    e = energy(x)
    f_c, v_t = contact_quantities(x)

    print(f"energy      : {e[0]:.4f} -> {e[-1]:.4f}, max increase {np.max(np.diff(e)):+.2e}")
    print(f"gap f_c     : min {f_c.min():+.2e}  (must stay >= 0)")
    print(f"v_tangential: {v_t[0]:+.3f} -> {v_t[-1]:+.3f}  (-> 0 means the disc rolls)")

    fig, axes = plt.subplots(4, 1, figsize=(7, 10))
    xs = np.linspace(x[0, :].min() - R, x[0, :].max() + R, 400)
    axes[0].plot(xs, A*np.sin(K*xs), "k-", lw=1, label="floor")
    axes[0].plot(x[0, :], x[1, :], label="disc centre")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].axis("equal")
    axes[1].plot(t, v_t); axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_ylabel("tangential contact velocity")
    axes[2].plot(t, f_c); axes[2].axhline(0.0, color="k", lw=0.5)
    axes[2].set_ylabel("gap f_c")
    axes[3].plot(t, e); axes[3].set_ylabel("energy"); axes[3].set_xlabel("t")
    for ax in axes:
        ax.grid(True)
    axes[0].legend()
    plt.tight_layout()

    
    _anim = animate(t, x)
    plt.show()


if __name__ == "__main__":
    main()
