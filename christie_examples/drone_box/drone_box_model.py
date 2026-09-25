import casadi as ca
import numpy as np

import nosnoc

from dataclasses import dataclass, field


# ------------------------------------------------------------------ ellipse-obstacle geometry
# Both obstacles are boxes in the original problem, approximated here by inscribed ellipses

def ellipse_params_bottom(cfg):
    cx = 0.5 * (cfg.x_min_b + cfg.x_max_b)
    a = 0.5 * (cfg.x_max_b - cfg.x_min_b)
    cz = 0.0
    b = cfg.z_max_b
    return cx, cz, a, b


def ellipse_params_top(cfg):
    cx = 0.5 * (cfg.x_min_b2 + cfg.x_max_b2)
    a = 0.5 * (cfg.x_max_b2 - cfg.x_min_b2)
    cz = 0.5 * (cfg.z_min_b2 + cfg.z_max_b2)
    b = 0.5 * (cfg.z_max_b2 - cfg.z_min_b2)
    return cx, cz, a, b


def _outside_ellipse_expr(px, pz, cx, cz, a, b):
    """>=0 outside (or on) the ellipse boundary, <0 inside."""
    return ((px - cx) / a) ** 2 + ((pz - cz) / b) ** 2 - 1.0


# ------------------------------------------------------------------ configuration

@dataclass
class DroneBoxConfig:
    nx: int = 5          # state: (p_x, p_z, pitch, b_x, b_z) -- drone pose + box position
    nu: int = 2          # controls: (f_left, f_right) rotor thrusts

    mass_d: float = 0.5  # drone mass [kg]
    mass_b: float = 0.1  # box mass [kg]
    inertia: float = 0.04
    d: float = 0.2                     # distance from the drone's center of mass to each rotor
    gravity: float = 9.81

    cable_length: float = 0.43         # length of the cable connecting the drone to the box

    # bottom obstacle (rests on the ground: z in [0, z_max_b])
    x_min_b: float = -2.0
    x_max_b: float = 2.0
    z_max_b: float = 0.5

    # top obstacle (floats: z in [z_min_b2, z_max_b2])
    x_min_b2: float = -2.0
    x_max_b2: float = 2.0
    z_min_b2: float = 0.825
    z_max_b2: float = 2.0


@dataclass
class DroneBoxOCPConfig:
    """
    Discretization, cost, initial/target state and solver settings for the tracking OCP.
    """
    sampling_time: float = 0.12
    N_stages: int = 50           # control horizon length; T = N_stages * sampling_time = 6.0 s
    n_s: int = 2                  # Runge-Kutta stages per finite element
    N_finite_elements: int = 1    # finite elements per control stage -- see the module docstring

    max_thrust: float = 15.0      # each rotor thrust is constrained to [0, max_thrust]

    # Both contacts (drone-ground, box-ground, cable) are modeled as perfectly inelastic (e = 0)
    restitution: float = 0.0

    # Running, terminal and control-effort cost weights.
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 100.0, 100.0]))
    Q_terminal: np.ndarray = field(default_factory=lambda: np.diag([0.0, 10.0, 10.0, 400.0, 400.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.05]))

    x_init: np.ndarray = field(default_factory=lambda: np.array([-3.0, 0.0, 0.0, -2.7, 0.0]))
    v_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    u_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # Final drone p_x is irrelevant (Q_terminal's first entry is 0) -- only the box's final
    # position/pitch/velocity are actually targeted; 0.0 there is just a placeholder.
    x_final: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 3.0, 0.0]))
    v_final: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    # Shape of the initial-guess/tracking trajectory through the gap
    wall_margin: float = 0.05
    climb_frac: float = 0.2
    land_frac: float = 0.2
    guess_trail_mode: str = "diagonal"  # "diagonal" (fits the gap) or "level" (does not, kept for comparison)

    # `nosnoc` homotopy solver settings.
    homotopy_update_slope: float = 0.2
    N_homotopy: int = 15
    complementarity_tol: float = 1e-6
    print_level: int = 0


# ------------------------------------------------------------------ initial guess / reference

def build_initial_guess_through_gap(cfg: DroneBoxConfig, ocp_cfg: DroneBoxOCPConfig, N):
    """
    Kinematic guess (positions only -- not meant to satisfy the dynamics)

    Returns an (N+1, nx) array; only columns 0,1 (drone p_x,p_z) and 3,4 (box b_x,b_z) are set,
    column 2 (pitch) stays 0.
    """
    x_guess = np.zeros((N + 1, cfg.nx))

    px0, pz0 = ocp_cfg.x_init[0], ocp_cfg.x_init[1]
    bx0, bz0 = ocp_cfg.x_init[3], ocp_cfg.x_init[4]
    bx_f, bz_f = ocp_cfg.x_final[3], ocp_cfg.x_final[4]

    cable_length = cfg.cable_length
    gap_height = cfg.z_min_b2 - cfg.z_max_b
    wall_margin = ocp_cfg.wall_margin

    if ocp_cfg.guess_trail_mode == "level":
        v_offset = 0.0
        h_offset = cable_length
        z_drone_cruise = 0.5 * (cfg.z_max_b + cfg.z_min_b2)
    elif ocp_cfg.guess_trail_mode == "diagonal":
        v_offset = min(gap_height - 2.0 * wall_margin, 0.95 * cable_length)
        v_offset = max(v_offset, 0.0)
        h_offset = np.sqrt(max(cable_length**2 - v_offset**2, 0.0))
        box_z_cruise = cfg.z_max_b + wall_margin
        z_drone_cruise = box_z_cruise + v_offset
    else:
        raise ValueError(f"Unknown ocp_cfg.guess_trail_mode: {ocp_cfg.guess_trail_mode!r}")

    # Drone's own end p_x so the box, trailing by h_offset, lands on bx_f.
    px_f = bx_f + h_offset

    N_climb = max(1, int(ocp_cfg.climb_frac * N))
    N_land = max(1, int(ocp_cfg.land_frac * N))
    N_cruise = N - N_climb - N_land
    if N_cruise < 1:
        raise ValueError("Horizon too short for climb + cruise + land phases.")

    def ease(s):
        return (1.0 - np.cos(np.pi * np.clip(s, 0.0, 1.0))) / 2.0

    for k in range(N + 1):
        if k <= N_climb:
            s = k / N_climb
            drone_px = px0
            drone_pz = ease(s) * z_drone_cruise
        elif k <= N_climb + N_cruise:
            s = (k - N_climb) / N_cruise
            drone_px = px0 + ease(s) * (px_f - px0)
            drone_pz = z_drone_cruise
        else:
            s = (k - N_climb - N_cruise) / N_land
            drone_px = px_f
            drone_pz = (1.0 - ease(s)) * z_drone_cruise

        x_guess[k, 0] = drone_px
        x_guess[k, 1] = drone_pz

        # The box's target here is always "trailing the drone on a taut diagonal cable"
        # (trail_bx/trail_bz), evaluated at THIS knot's own drone position -- since the drone's
        # own trajectory is already continuous across the climb/cruise/land phase boundaries,
        # this target is continuous too. Only during climb/land do we blend the box smoothly
        # between its real endpoint (x_init/x_final) and that trailing target, so the guess still
        # starts/ends exactly on the actual boundary conditions.
        trail_bx = drone_px - h_offset
        # Floored at the ground: during climb/land the drone hasn't reached z_drone_cruise yet,
        # so drone_pz - v_offset can dip below 0, which would target the box underground.
        trail_bz = max(drone_pz - v_offset, 0.0)

        if k <= N_climb:
            s = k / N_climb
            box_px = (1.0 - ease(s)) * bx0 + ease(s) * trail_bx
            box_pz = (1.0 - ease(s)) * bz0 + ease(s) * trail_bz
        elif k <= N_climb + N_cruise:
            box_px = trail_bx
            box_pz = trail_bz
        else:
            s = (k - N_climb - N_cruise) / N_land
            box_px = (1.0 - ease(s)) * trail_bx + ease(s) * bx_f
            box_pz = (1.0 - ease(s)) * trail_bz + ease(s) * bz_f

        x_guess[k, 3] = box_px
        x_guess[k, 4] = box_pz

    return x_guess

# ------------------------------------------------------------------ model

def build_drone_box_model(cfg: DroneBoxConfig, ocp_cfg: DroneBoxOCPConfig, x_ref_full: np.ndarray):
    """
    Build the "drone towing a box through a gap" problem as a `nosnoc.model.Cls`.

    Physics:
        state   x = (q, v), q = (p_x, p_z, pitch, b_x, b_z),
                            v = (v_x, v_z, v_pitch, v_bx, v_bz)   [n_q = n_v = 5]
        control u = (f_left, f_right)                             [rotor thrusts >= 0]

        M(q) v_dot = f_v(q, v, u) + J_normal(q) @ lambda_normal

    with M = diag(mass_d, mass_d, inertia, mass_b, mass_b) constant, and three unilateral
    contacts stacked in `f_c = (f_c_ground_drone, f_c_ground_box, f_c_cable)`:

      * f_c_ground_drone(q) = p_z >= 0, f_c_ground_box(q) = b_z >= 0
      * f_c_cable(q) = cable_length - ||(p_x,p_z) - (b_x,b_z)|| >= 0 

    Two box-shaped obstacles must be avoided by *both* bodies (4 inequalities total).
    """
    nx, nu = cfg.nx, cfg.nu

    # ------------------------------------------------------------------ states / controls
    q = ca.SX.sym("q", nx)   # (p_x, p_z, pitch, b_x, b_z)
    v = ca.SX.sym("v", nx)   # (v_x, v_z, v_pitch, v_bx, v_bz)
    x = ca.vertcat(q, v)
    u = ca.SX.sym("u", nu)   # (f_left, f_right)
    pitch = q[2]
    f_left, f_right = u[0], u[1]

    # ------------------------------------------------------------------ rigid body dynamics
    M = np.diag([cfg.mass_d, cfg.mass_d, cfg.inertia, cfg.mass_b, cfg.mass_b])

    Fx_thrust = -(f_left + f_right) * ca.sin(pitch)
    Fz_thrust = (f_left + f_right) * ca.cos(pitch)
    torque_thrust = (f_right - f_left) * cfg.d

    # Generalized force excluding contact forces. Only the drone is thrust-actuated
    # and subject to gravity through its own mass; the box is affected only by gravity and the
    # cable/ground contact forces.
    f_v = ca.vertcat(
        Fx_thrust,
        Fz_thrust - cfg.mass_d * cfg.gravity,
        torque_thrust,
        0.0,
        -cfg.mass_b * cfg.gravity,
    )

    # ------------------------------------------------------------------ contact gap functions
    f_c_ground_drone = q[1]  # p_z >= 0
    f_c_ground_box = q[4]    # b_z >= 0

    diff = q[0:2] - q[3:5]
    dist = ca.sqrt(ca.dot(diff, diff) + 1e-12)  # regularized distance, matches the original model
    f_c_cable = cfg.cable_length - dist

    f_c = ca.vertcat(f_c_ground_drone, f_c_ground_box, f_c_cable)  # n_c = 3

    # ------------------------------------------------------------------ obstacle-avoidance path constraints
    cx_b, cz_b, a_b, b_b = ellipse_params_bottom(cfg)
    cx_t, cz_t, a_t, b_t = ellipse_params_top(cfg)
    g_path = ca.vertcat(
        _outside_ellipse_expr(q[0], q[1], cx_b, cz_b, a_b, b_b),  # drone vs. bottom obstacle
        _outside_ellipse_expr(q[3], q[4], cx_b, cz_b, a_b, b_b),  # box vs. bottom obstacle
        _outside_ellipse_expr(q[0], q[1], cx_t, cz_t, a_t, b_t),  # drone vs. top obstacle
        _outside_ellipse_expr(q[3], q[4], cx_t, cz_t, a_t, b_t),  # box vs. top obstacle
    )

    # ------------------------------------------------------------------ cost
    x_ref = ca.SX.sym("x_ref", nx)
    u_hover = 0.5 * cfg.mass_d * cfg.gravity * ca.DM.ones(nu)  # hover thrust, split across rotors
    f_q = (q - x_ref).T @ ocp_cfg.Q @ (q - x_ref) + (u - u_hover).T @ ocp_cfg.R @ (u - u_hover)

    # Terminal cost: track the final target in *both* position and velocity.
    q_final = ca.DM(ocp_cfg.x_final)
    v_final = ca.DM(ocp_cfg.v_final)
    f_q_T = (q - q_final).T @ ocp_cfg.Q_terminal @ (q - q_final) \
        + (v - v_final).T @ ocp_cfg.Q_terminal @ (v - v_final)

    # ------------------------------------------------------------------ bounds / initial state
    lbu = np.zeros(nu)
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
        g_path=g_path, lbg_path=np.zeros(4), ubg_path=np.inf * np.ones(4),
        p_time_var=x_ref, p_time_var_val=x_ref_full[0, :],  # overwritten per stage before solving
        f_q=f_q,
        f_q_T=f_q_T,
        name="drone_box_gap",
    )
    return model
