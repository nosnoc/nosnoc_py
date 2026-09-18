"""
Contact geometry of a planar ellipse pinched between two balls.

This module is deliberately free of any CLS specific code: it only provides the kinematics of the
two ball-ellipse contacts (gap functions, contact frames and contact Jacobians) and the scene
parameters. A first order (projected dynamical system) version of the same scene can therefore
import it unchanged and only differ in how the contact forces enter the dynamics.

Generalized coordinates, in this order::

    q = (x_e, y_e, theta, x_1, y_1, x_2, y_2)

i.e. the ellipse pose followed by the centers of the two balls, so $n_q = 7$ and $n_c = 2$.
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np

GRAVITY = 9.81


@dataclass
class GraspConfig:
    """Every parameter of the scene and of the impedance controller in one place."""
    # Geometry and inertia.
    a: float = 0.1          # Semi axis of the ellipse along its body x axis.
    b: float = 0.4          # Semi axis of the ellipse along its body y axis.
    m_ellipse: float = 1.0
    r_ball: float = 0.1
    m_ball: float = 0.2
    mu: float = 0.6          # Coefficient of friction at both ball-ellipse contacts.

    # Impedance controller. `impedance = False` makes u a force on each ball instead.
    impedance: bool = True
    k_impedance: float = 500.0   # Stiffness [N/m].
    d_impedance: float = 20.0    # Damping [Ns/m], 2*sqrt(k*m_ball) is critical.

    # Task.
    initial_gap: float = 0.5  # Clearance between each ball and the ellipse at t = 0 [m].
    h_lift: float = 0.3      # How far the ellipse should be lifted [m].
    T: float = 2.0             # Horizon [s].

    @property
    def J_ellipse(self) -> float:
        """Moment of inertia of a homogeneous elliptic disc about its center."""
        return self.m_ellipse*(self.a**2 + self.b**2)/4

    @property
    def inertia_diag(self) -> np.ndarray:
        m_b = self.m_ball
        return np.array([self.m_ellipse, self.m_ellipse, self.J_ellipse, m_b, m_b, m_b, m_b])

    @property
    def d_contact(self) -> float:
        """Half distance between the ball centers when both just touch the level ellipse."""
        return self.a + self.r_ball

    @property
    def x0(self) -> np.ndarray:
        """
        Ellipse level at the origin, both balls held `initial_gap` clear of it, everything at rest.

        Starting out of contact is what makes the problem a grasp rather than a hold: nothing
        supports the ellipse, so it free-falls until the fingers close on it, and the catch is a
        plastic impact. Set `initial_gap = 0.0` to start in resting contact instead.
        """
        d = self.d_contact + self.initial_gap
        return np.array([0.0, 0.0, 0.0, d, 0.0, -d, 0.0] + [0.0]*7)

    @property
    def u0(self) -> np.ndarray:
        """
        Initial guess for the control.

        In impedance mode `u` is a *desired position*, so it must start at the ball positions.
        Leaving it at the default of zero would place both targets in the middle of the ellipse and
        pull the fingers straight through it.
        """
        if not self.impedance:
            return np.zeros(4)
        d = self.d_contact + self.initial_gap
        return np.array([d, 0.0, -d, 0.0])

    @property
    def x_ref(self) -> np.ndarray:
        """
        Terminal target: the ellipse lifted by `h_lift`, at rest, with the fingers closed on it.

        The balls are referenced at the *contact* distance rather than at their initial clearance,
        so the tracking cost asks them to close in and stay there. Referencing them where they
        started would ask them to let go again.
        """
        d = self.d_contact
        x_ref = np.array([0.0, self.h_lift, 0.0, d, self.h_lift, -d, self.h_lift] + [0.0]*7)
        return x_ref


def rot(theta):
    """Planar rotation matrix, CasADi or numpy depending on the argument."""
    if isinstance(theta, (ca.SX, ca.MX, ca.DM)):
        c, s = ca.cos(theta), ca.sin(theta)
        return ca.vertcat(ca.horzcat(c, -s), ca.horzcat(s, c))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def ball_slice(q, i):
    """Indices are easy to get wrong, so the ball blocks are named once here."""
    return q[3 + 2*i:5 + 2*i]


def gap(c, theta, p, cfg: GraspConfig):
    r"""
    Gap function of one ball against the ellipse, as the level set of the *inflated* ellipse.

    With $d = p - c$ and $\tilde{e} = R(\theta)^\top d$ the body frame offset,

    $$f_c = (\tilde{e}_x/(a+r))^2 + (\tilde{e}_y/(b+r))^2 - 1,$$

    which is positive exactly when the ball center lies outside the ellipse grown by the ball
    radius. Two approximations are folded in here and both are deliberate:

    * the exact offset curve of an ellipse is not an ellipse, so growing the semi axes by `r` is
      only an approximation of the Minkowski sum, and
    * the level set is not a distance, a 5 cm separation gives $f_c \approx 0.21$.

    Neither matters for the complementarity, which only uses the *sign* of `f_c`. The magnitude
    would matter for the friction cone, which is why the contact Jacobians below are built
    explicitly rather than differentiated from this expression, see `contact_terms`.
    """
    e_local = rot(theta).T@(p - c)
    return (e_local[0]/(cfg.a + cfg.r_ball))**2 + (e_local[1]/(cfg.b + cfg.r_ball))**2 - 1


def contact_frame(c, theta, p, cfg: GraspConfig):
    """
    Unit normal, tangent and contact point of one ball-ellipse contact.

    The normal is the normalized gradient of the level set with respect to the ball center, so it
    points away from the ellipse, i.e. in the direction of increasing gap.
    """
    grad = ca.jacobian(gap(c, theta, p, cfg), p).T
    n = grad/ca.norm_2(grad)
    t = ca.vertcat(-n[1], n[0])
    x_c = p - cfg.r_ball*n
    return n, t, x_c


def contact_terms(q, v, cfg: GraspConfig):
    r"""
    Gap functions and contact Jacobians of both contacts.

    Returns `(f_c, J_normal, J_tangent)` with `f_c` of size 2 and both Jacobians of size 7x2.

    The Jacobians are built explicitly instead of being left to `jacobian(f_c, q).T`, and that is
    essential rather than cosmetic. `nosnoc` writes the Coulomb condition directly on the
    multipliers, $\sum \lambda_{\mathrm{t}} \le \mu\lambda_{\mathrm{n}}$, see
    `dcs/cls.py::_friction_equations`. That is only Coulomb friction if the multipliers are genuine
    forces, which in turn requires the Jacobians to be *power dual*: their transposes must map the
    generalized velocity to the physical relative velocity at the contact, in m/s. The gradient of
    the level set above carries an arbitrary scale instead, which would silently rescale `mu`.

    So with $n$ the unit normal, $t$ the tangent and $x_c$ the contact point, the relative velocity
    of the ball with respect to the material point of the ellipse at $x_c$ is

    $$v_{\mathrm{rel}} = v_i - \big(\dot{c} + \dot{\theta}\, S(x_c - c)\big),
      \qquad S(w) = (-w_y, w_x),$$

    and the columns are the gradients of $n^\top v_{\mathrm{rel}}$ and $t^\top v_{\mathrm{rel}}$
    with respect to `v`. The `theta` row of the tangential column is the lever arm of the contact
    point, which is what lets the fingers spin the ellipse.

    Note that these columns are *not* unit vectors, as they act on three bodies at once, so
    `nosnoc.model.Cls` will warn about `D_tangent`. The warning assumes a single body contact where
    the column is a plain direction; for a power dual multi body Jacobian it is a false positive.
    """
    c, theta = q[0:2], q[2]
    v_c, dtheta = v[0:2], v[2]

    f_c, cols_n, cols_t = [], [], []
    for i in range(2):
        p_i, v_i = ball_slice(q, i), ball_slice(v, i)
        n, t, x_c = contact_frame(c, theta, p_i, cfg)
        w = x_c - c
        v_rel = v_i - (v_c + dtheta*ca.vertcat(-w[1], w[0]))
        f_c.append(gap(c, theta, p_i, cfg))
        cols_n.append(ca.jacobian(n.T@v_rel, v).T)
        cols_t.append(ca.jacobian(t.T@v_rel, v).T)
    return ca.vertcat(*f_c), ca.horzcat(*cols_n), ca.horzcat(*cols_t)


def ellipse_outline(c, theta, cfg: GraspConfig, n_pts=200):
    """Outline of the ellipse as `(x, y)` arrays, for plotting."""
    s = np.linspace(0, 2*np.pi, n_pts)
    pts = rot(theta)@np.vstack([cfg.a*np.cos(s), cfg.b*np.sin(s)])
    return pts[0] + c[0], pts[1] + c[1]


def circle_outline(center, radius, n_pts=100):
    """Outline of a circle as `(x, y)` arrays, following `disc_around_obstacle._circle`."""
    s = np.linspace(0, 2*np.pi, n_pts)
    return radius*np.cos(s) + center[0], radius*np.sin(s) + center[1]
