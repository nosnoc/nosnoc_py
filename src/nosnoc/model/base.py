from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real

import casadi as ca
import numpy as np

from ..dims import Dims

class BaseDims(Dims):
    def __init__(self):
        super().__init__(None)
        self.n_x = 0
        self.n_u = 0
        self.n_z = 0
        self.n_v_global = 0
        self.n_p_global = 0
        self.n_p_time_var = 0
        self.n_g_path = 0
        self.n_g_terminal = 0
        self.n_g_comp = 0

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

    def __init__(
            self,
            x: ca.SX,
            lbx: Optional[np.ndarray|Real] = None,
            ubx: Optional[np.ndarray|Real] = None,
            x0: Optional[np.ndarray|Real] = None,
            z: Optional[ca.SX] = None,
            z0: Optional[np.ndarray|Real] = None,
            lbz: Optional[np.ndarray|Real] = None,
            ubz: Optional[np.ndarray|Real] = None,
            g_z: Optional[ca.SX] = None,
            u: Optional[ca.SX] = None,
            lbu: Optional[np.ndarray|Real] = None,
            ubu: Optional[np.ndarray|Real] = None,
            u0: Optional[np.ndarray|Real] = None,
            v_global: Optional[ca.SX] = None,
            v0_global: Optional[np.ndarray|Real] = None,
            lbv_global: Optional[np.ndarray|Real] = None,
            ubv_global: Optional[np.ndarray|Real] = None,
            p_global: Optional[ca.SX] = None,
            p_global_val: Optional[np.ndarray|Real] = None,
            p_time_var: Optional[ca.SX] = None,
            p_time_var_val: Optional[np.ndarray|Real] = None,
            f_q: Optional[ca.SX] = None,
            f_q_T: Optional[ca.SX] = None,
            lsq_x: Optional[np.ndarray|Real] = None,
            x_ref_val: Optional[np.ndarray|Real] = None,
            lsq_u: Optional[np.ndarray|Real] = None,
            u_ref_val: Optional[np.ndarray|Real] = None,
            lsq_T: Optional[np.ndarray|Real] = None,
            x_ref_end_val: Optional[np.ndarray|Real] = None,
            g_path: Optional[ca.SX] = None,
            lbg_path: Optional[np.ndarray|Real] = None,
            ubg_path: Optional[np.ndarray|Real] = None,
            g_terminal: Optional[ca.SX] = None,
            lbg_terminal: Optional[np.ndarray|Real] = None,
            ubg_terminal: Optional[np.ndarray|Real] = None,
            G_path: Optional[ca.SX] = None,
            H_path: Optional[ca.SX] = None,
            name: str = "nosnoc_model",
    ):
        self.dims = BaseDims()
        # Vectors
        self.x = x; self.lbx = lbx; self.ubx = ubx; self.x0 = x0
        self._populate_vectors("x", [("lbx", -np.inf), ("ubx", np.inf), ("x0", 0.0)])
        self.dims.n_x = self.x.size(1)
        self.z = z; self.lbz = lbz; self.ubz = ubz; self.z0 = z0
        self._populate_vectors("z", [("lbz", -np.inf), ("ubz", np.inf), ("z0", 0.0)])
        self.dims.n_z = self.z.size(1)
        self.g_z = g_z; self._populate_scalar("g_z", [])
        self.u = u; self.lbu = lbu; self.ubu = ubu; self.u0 = u0
        self._populate_vectors("u", [("lbu", -np.inf), ("ubu", np.inf), ("u0", 0.0)])
        self.dims.n_u = self.u.size(1)
        self.v_global = v_global; self.lbv_global = lbv_global; self.ubv_global = ubv_global; self.v0_global = v0_global
        self._populate_vectors("v_global", [("lbv_global", -np.inf), ("ubv_global", np.inf), ("v0_global", 0.0)])
        self.dims.n_v_global = self.v_global.size(1)
        self.p_global = p_global; self.p_global_val = p_global_val
        self._populate_vectors("p_global", [("p_global_val", 0.0)])
        self.dims.n_p_global = self.p_global.size(1)
        self.p_time_var = p_time_var; self.p_time_var_val = p_time_var_val
        self._populate_vectors("p_time_var", [("p_time_var_val", 0.0)])
        self.dims.n_p_time_var = self.p_time_var.size(1)
        self.p = ca.vertcat(self.p_global, self.p_time_var)
        self.g_path = g_path; self.lbg_path = lbg_path; self.ubg_path = ubg_path
        self._populate_vectors("g_path", [("lbg_path", -np.inf), ("ubg_path", np.inf)])
        self.dims.n_g_path = self.g_path.size(1)
        self.g_terminal = g_terminal; self.lbg_terminal = lbg_terminal; self.ubg_terminal = ubg_terminal
        self._populate_vectors("g_terminal", [("lbg_terminal", 0.0), ("ubg_terminal", 0.0)])
        self.dims.n_g_terminal = self.g_terminal.size(1)

        self.G_path = G_path; self._populate_vectors("G_path")
        self.H_path = H_path; self._populate_vectors("H_path")
        self.dims.n_g_comp = self.G_path.size(1)

        # Scalars
        self.f_q = f_q; self._populate_scalar("f_q", 0.0)
        self.f_q_T = f_q_T; self._populate_scalar("f_q_T", 0.0)

        # TODO(@anton) implement this reasonably.
        self.lsq_x = lsq_x
        self.x_ref_val = x_ref_val
        self.lsq_u = lsq_u
        self.u_ref_val = u_ref_val
        self.lsq_T = lsq_T
        self.x_ref_end_val = x_ref_end_val

        self.name = name

    def _populate_vectors(self, sym: str, vec_init_list = []):
        """
        Take a symbolic vector and a list of its related numerical vectors and populate them all.
        """
        if getattr(self, sym) is None: # if empty sym throw away
            setattr(self, sym, ca.SX([]))
        n,m = getattr(self, sym).size()
        for (vec, init) in vec_init_list:
            if getattr(self, vec) is None:
                setattr(self, vec,init*np.ones(n))
            elif isinstance(getattr(self, vec), Real):
                setattr(self, vec, getattr(self, vec)*np.ones(n))
            elif np.array(getattr(self, vec)).shape[0] != n:
                raise RuntimeError("Dimension missmatch in model creation") # TODO(@anton) make this error more traceable

    def _populate_scalar(self, sym: str, default=0.0):
        """
        Take a symbolic scalar and populate it with a default if necessary
        """
        if getattr(self, sym) is None:
            setattr(self, sym, ca.SX(default))
