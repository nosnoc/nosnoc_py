import casadi as ca
import numpy as np

import nosnoc

from dataclasses import dataclass, field


def build_reference_trajectory(N, nx, x_center=0.0, ellipse_center_z=1.0,
                                ellipse_a=1.0, ellipse_b=0.25,
                                climb_frac=0.2, land_frac=0.2):
    """
    Reference trajectory for the drone-on-a-cable example: climb straight up, 
    trace one loop of an ellipse, then descend straight back down.
    """
    x_ref = np.zeros((N + 1, nx))

    N_climb = max(1, int(climb_frac * N))
    N_land = max(1, int(land_frac * N))
    N_ellipse = N - N_climb - N_land
    if N_ellipse < 1:
        raise ValueError("Horizon too short for climb + ellipse + land phases.")

    z_bottom = ellipse_center_z - ellipse_b

    def ease(s):
        """Smooth (cosine) 0 -> 1 easing, avoids a velocity jump at the start/end of a phase."""
        return (1.0 - np.cos(np.pi * np.clip(s, 0.0, 1.0))) / 2.0

    # Phase 1: climb straight up (p_x constant, p_z eased 0 -> z_bottom).
    for k in range(N_climb + 1):
        s = k / N_climb
        x_ref[k, 0] = x_center
        x_ref[k, 1] = ease(s) * z_bottom

    # Phase 2: one full ellipse loop, starting/ending at the bottom (angle = -pi/2) so it
    # connects seamlessly to the end of the climb.
    for i in range(1, N_ellipse + 1):
        k = N_climb + i
        s = i / N_ellipse
        angle = -np.pi / 2 + 2.0 * np.pi * s
        x_ref[k, 0] = x_center + ellipse_a * np.cos(angle)
        x_ref[k, 1] = ellipse_center_z + ellipse_b * np.sin(angle)

    # Phase 3: descend straight back down (p_x constant, p_z eased z_bottom -> 0), the mirror
    # image of the climb.
    for i in range(1, N_land + 1):
        k = N_climb + N_ellipse + i
        if k > N:
            break
        s = i / N_land
        x_ref[k, 0] = x_center
        x_ref[k, 1] = (1.0 - ease(s)) * z_bottom

    return x_ref


@dataclass
class DroneCableConfig:
    """Physical parameters of the planar drone and the cable that tethers it to the ground."""
    nx: int = 3         # state : (p_x, p_z, pitch)
    nu: int = 2         # controls: (f_left, f_right) rotor thrusts

    mass: float = 0.5   # drone mass [kg]
    inertia: float = 0.04
    d: float = 0.2                     # distance from the center of mass to each rotor
    gravity: float = 9.81

    cable_length: float = 1.2          # length of the cable tethering the drone to the ground
    cable_origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

@dataclass
class DroneCableOCPConfig:
    """
    Discretization, cost and solver settings for the tracking OCP.
    """
    sampling_time: float = 0.15
    N_stages: int = 50           # control horizon length; T = N_stages * sampling_time = 7.5 s
    n_s: int = 2                 # Runge-Kutta stages per finite element
    N_finite_elements: int = 2   # finite elements per control stage

    max_thrust: float = 15.0     # each rotor thrust is constrained to [0, max_thrust]

    # Both contacts (ground and cable) are modeled as perfectly inelastic (e = 0)
    restitution: float = 0.0

    # Running and terminal cost weights
    Q: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 1.0]))
    Q_terminal: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01]))

    x_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    v_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    u_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # Reference trajectory shape: climb -> one ellipse loop -> descend,
    # see `drone_cable_reference.py::build_reference_trajectory`.
    ellipse_center_z: float = 1.0
    ellipse_a: float = 1.0
    ellipse_b: float = 0.25
    climb_frac: float = 0.2
    land_frac: float = 0.2

    # `nosnoc` homotopy solver settings
    homotopy_update_slope: float = 0.2
    N_homotopy: int = 12
    complementarity_tol: float = 1e-6
    print_level: int = 3

def build_drone_cable_model(cfg: DroneCableConfig, ocp_cfg: DroneCableOCPConfig, x_ref_T_val: np.ndarray):
    """
    Build Planar "drone tethered to the ground by a cable" as a `nosnoc.model.Cls`
    
    Physics:
        state   x = (q, v),  q = (p_x, p_z, pitch),  v = (v_x, v_z, v_pitch)   [n_q = n_v = 3]
        control u = (f_left, f_right)                                        [rotor thrusts >= 0]

        M(q) v_dot = f_v(q, v, u) + J_normal(q) @ lambda_normal

    with M = diag(mass, mass, inertia) constant, and two unilateral contacts stacked in
    `f_c = (f_c_ground, f_c_cable)`

    """
    nx, nu = cfg.nx, cfg.nu

    # ------------------------------------------------------------------ states / controls
    q = ca.SX.sym("q", nx)   # (p_x, p_z, pitch)
    v = ca.SX.sym("v", nx)   # (v_x, v_z, v_pitch)
    x = ca.vertcat(q, v)
    u = ca.SX.sym("u", nu)   # (f_left, f_right)
    pitch = q[2]
    f_left, f_right = u[0], u[1]

    # ------------------------------------------------------------------ rigid body dynamics
    M = np.diag([cfg.mass, cfg.mass, cfg.inertia])

    Fx_thrust = -(f_left + f_right) * ca.sin(pitch)
    Fz_thrust = (f_left + f_right) * ca.cos(pitch)
    torque_thrust = (f_right - f_left) * cfg.d

    # Generalized force excluding contact forces
    f_v = ca.vertcat(
        Fx_thrust,
        Fz_thrust - cfg.mass * cfg.gravity,
        torque_thrust,
    )

    # ------------------------------------------------------------------ contact gap functions
    f_c_ground = q[1]  # p_z >= 0

    diff = q[0:2] - ca.DM(cfg.cable_origin)
    dist = ca.sqrt(ca.dot(diff, diff) + 1e-12)  # regularized distance
    f_c_cable = cfg.cable_length - dist  # cable_length - dist >= 0

    f_c = ca.vertcat(f_c_ground, f_c_cable)  # n_c = 2: [ground, cable]

    # ------------------------------------------------------------------ reference-tracking cost
    # Time-varying stage reference: one numeric value per control stage
    x_ref = ca.SX.sym("x_ref", nx)

    # NOTE: the cost only tracks the position/pitch reference `q` (Q, Q_terminal are 3x3);
    # velocity is not penalized directly, only regularized indirectly through the control-effort term below.
    u_hover = 0.5 * cfg.mass * cfg.gravity * ca.DM.ones(nu)  # hover thrust, split across rotors
    f_q = (q - x_ref).T @ ocp_cfg.Q @ (q - x_ref) + (u - u_hover).T @ ocp_cfg.R @ (u - u_hover)
    # Terminal reference is a numeric constant, see the module docstring.
    f_q_T = (q - ca.DM(x_ref_T_val)).T @ ocp_cfg.Q_terminal @ (q - ca.DM(x_ref_T_val))

    # ------------------------------------------------------------------ bounds / initial state
    lbu = np.full(nu, -ocp_cfg.max_thrust)
    ubu = np.full(nu, ocp_cfg.max_thrust)
    x0 = np.concatenate([ocp_cfg.x_init, ocp_cfg.v_init])

    model = nosnoc.model.Cls(
        x=x,
        u=u, lbu=lbu, ubu=ubu, u0=ocp_cfg.u_init,
        x0=x0,
        M=M,
        f_v=f_v,
        f_c=f_c,
        e=ocp_cfg.restitution,
        mu=0.0,  # frictionless contacts
        p_time_var=x_ref, p_time_var_val=np.zeros(nx),   # overwritten per stage before solving
        f_q=f_q,
        f_q_T=f_q_T,
        name="drone_ground_cable",
    )
    return model
