from abc import ABC, abstractmethod
import casadi as ca
import numpy as np


class Base(ABC):
    r"""
    Base class for all ``nosnoc`` models. It contains shared properties such as the state, user algebraics, controls etc.
    It also contains the fields used to populate an Optimal Control problem such as Lagrange and Mayer cost terms,
    an interface for least squares costs as well as non-box path and terminal constraints.

    :param x:  casadi.SX: Differential state $x \in \mathbb{R}^{n_x}$
    :param lbx:  numpy.ndarray: $\underline{x} \in \mathbb{R}^{n_x}$, differential state lower bound.
    :param ubx:  numpy.ndarray: $\bar{x} \in \mathbb{R}^{n_x}$, differential state upper bound.
    :param x0:  numpy.ndarray: $x_0 \in \mathbb{R}^{n_x}$, initial differential state, also used to initialize all differential state variables in the resulting MPCC.

    :param z:  casadi.SX: User algebraics $z \in \mathbb{R}^{n_z}$
    :param z0:  numpy.ndarray: $z_0 \in \mathbb{R}^{n_z}$, used to initialize all user algebraic variables in the resulting MPCC.
    :param lbz:  numpy.ndarray: $\underline{z} \in \mathbb{R}^{n_z}$, user algebraic lower bound.
    :param ubz:  numpy.ndarray: $\bar{z} \in \mathbb{R}^{n_z}$, user algebraic upper bound.
    :param g_z:  casadi.SX: Constraint expression used to define the behavior of user algebraics.

    :param u:  casadi.SX: Controls $u \in \mathbb{R}^{n_u}$.
    :param lbu:  numpy.ndarray: $\underline{u} \in \mathbb{R}^{n_u}$, controls lower bound.
    :param ubu:  numpy.ndarray: $\bar{u} \in \mathbb{R}^{n_u}$, controls upper bound.
    :param u0:  numpy.ndarray: $u_0 \in \mathbb{R}^{n_u}$, used to initialize all control variables in the resulting MPCC.

    :param v_global:  casadi.SX: $\nu \in \mathbb{R}^{n_{\nu}}$ global variables (not time dependent).
    :param v0_global:  numpy.ndarray: $\nu_0 \in \mathbb{R}^{n_{\nu}}$, used to initialize all global variables in the resulting MPCC.
    :param lbv_global:  numpy.ndarray: $\underline{\nu} \in \mathbb{R}^{n_{\nu}}$, global variables lower bound.
    :param ubv_global:  numpy.ndarray: $\bar{\nu} \in \mathbb{R}^{n_{\nu}}$, global variables upper bound.

    :param p_global:  casadi.SX: Global parameters.
    :param p_global_val:  numpy.ndarray: Values for global parameters

    :param p_time_var:  casadi.SX: Time varying parameters which are considered to be constant over each control/integration interval.
    :param p_time_var_val:  numpy.ndarray: Values for time varying parameters.
    :param p:  casadi.SX: All model parameters

    :param f_q:  casadi.SX: Lagrange term cost.
    :param f_q_T:  casadi.SX: Mayer term cost.

    :param lsq_x:  cell: TODO describe
    :param x_ref:  casadi.SX:
    :param f_lsq_x:  casadi.SX:
    :param x_ref_val:  numpy.ndarray:
    :param lsq_u:  casadi.SX:
    :param u_ref:  casadi.SX:
    :param f_lsq_u:  casadi.SX:
    :param u_ref_val:  numpy.ndarray: vector
    :param lsq_T:  casadi.SX:
    :param x_ref_end:  casadi.SX:
    :param f_lsq_T:  casadi.SX:
    :param x_ref_end_val:  numpy.ndarray: vector

    :param g_path:  casadi.SX: Path constraints.
    :param lbg_path:  numpy.ndarray: Lower bound on path constraints.
    :param ubg_path:  numpy.ndarray: Upper bound on path constraints.

    :param g_terminal:  casadi.SX: Terminal constraints.
    :param lbg_terminal:  numpy.ndarray: Lower bound on path constraints.
    :param ubg_terminal:  numpy.ndarray: Upper bound on path constraints.

    :param G_path:  casadi.SX: One half of path complementarities.
    :param H_path:  casadi.SX: One half of path complementarities.

    :param dims:  struct: Dimensions struct, the contents of which depends on the subclass.
    """

    def __init__(self,
                 x: ca.SX,
                 lbx: Optional[np.ndarray] = None,
                 ubx: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None,
                 z: Optional[ca.SX] = None,
                 z0: Optional[np.ndarray] = None,
                 lbz: Optional[np.ndarray] = None,
                 ubz: Optional[np.ndarray] = None,
                 g_z: Optional[ca.SX] = None,
                 u: Optional[ca.SX] = None,
                 lbu: Optional[np.ndarray] = None,
                 ubu: Optional[np.ndarray] = None,
                 u0: Optional[np.ndarray] = None,
                 v_global: Optional[ca.SX] = None,
                 v0_global: Optional[np.ndarray] = None,
                 lbv_global: Optional[np.ndarray] = None,
                 ubv_global: Optional[np.ndarray] = None,
                 p_global: Optional[ca.SX] = None,
                 p_global_val: Optional[np.ndarray] = None,
                 p_time_var: Optional[ca.SX] = None,
                 p_time_var_val: Optional[np.ndarray] = None,
                 p: Optional[ca.SX] = None,
                 f_q: Optional[ca.SX] = None,
                 f_q_T: Optional[ca.SX] = None,
                 lsq_x: Optional[np.ndarray] = None,
                 x_ref_val: Optional[np.ndarray] = None,
                 lsq_u: Optional[np.ndarray] = None,
                 u_ref_val: Optional[np.ndarray] = None,
                 lsq_T: Optional[np.ndarray] = None,
                 x_ref_end_val: Optional[np.ndarray] = None,
                 g_path: Optional[ca.SX] = None,
                 lbg_path: Optional[np.ndarray] = None,
                 ubg_path: Optional[np.ndarray] = None,
                 g_terminal: Optional[ca.SX] = None,
                 lbg_terminal: Optional[np.ndarray] = None,
                 ubg_terminal: Optional[np.ndarray] = None,
                 G_path: Optional[ca.SX] = None,
                 H_path: Optional[ca.SX] = None,
                 ):
        pass
