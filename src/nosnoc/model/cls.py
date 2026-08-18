from .base import Base, BaseDims
from ..dims import Dims

from typing import Optional, List
from numbers import Real

import casadi as ca
import numpy as np



class ClsDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_q = 0 # Number of generalized coordinates.
        self.n_v = 0 # Number of generalized velocities, equal to n_q.
        self.n_c = 0 # Number of possible contacts.
        self.n_t = 0 # Number of tangential directions per contact (0 if frictionless).
        self.n_tangents = 0 # Total number of tangential multipliers, n_t*n_c.


class Cls(Base):
    r"""
    A system of rigid bodies with contacts and friction, i.e., a Complementarity Lagrangian System:

   

        
          $ q_dot = M(q) v_dot = f_v(q,v) + sum (J_n}^i lambda_n^i + J_t^i lambda_t^i) 
          
          $
                          
           0 &\le \lambda_{\mathrm{n}}^i \perp f_c^i(q) \ge 0 \\
           0 &= J_{\mathrm{n}}^i(q(t_s))^\top(v(t_s^+) + e^i v(t_s^-))
                \quad\mathrm{if}\ f_c^i(q(t_s)) = 0\ \mathrm{and}\ J_{\mathrm{n}}^i(q(t_s))^\top v(t_s^-) < 0
        \end{align*}

    with $i = 1\ldots n_c$. This model is discretized with the FESD-J method.

   

    Note:
        Friction is not yet implemented. Passing a nonzero coefficient of friction raises
        a `NotImplementedError`.

    """
    def __init__(self,
                 *,
                 q: Optional[ca.SX] = None, # Generalized coordinates, defaults to the first half of x.
                 v: Optional[ca.SX] = None, # Generalized velocities, defaults to the second half of x.
                 f_v: ca.SX, # Generalized force, $M(q)\dot{v} = f_v(x)\in\mathbb{R}^{n_q}$.
                 f_c: ca.SX, # Contact gap functions $f_c(q)\in\mathbb{R}^{n_c}$.
                 mu: Optional[float|List[float]|np.ndarray] = None, # Coefficient(s) of friction.
                 e: float|List[float]|np.ndarray, # Coefficient(s) of restitution in $[0,1]$.
                 M: Optional[ca.SX|np.ndarray] = None, # Generalized inertia matrix, may depend on $q$.
                 inv_M: Optional[ca.SX|np.ndarray] = None, # User provided inverse of the inertia matrix.
                 J_normal: Optional[ca.SX] = None, # Normal contact Jacobian, computed from f_c if omitted.
                 J_tangent: Optional[ca.SX] = None, # Tangent contact Jacobian, required for Conic friction.
                 # Polyhedral tangent Jacobian, required for Polyhedral friction.
                 # For every column $D_i$, $-D_i$ must also be a column of $D$.
                 D_tangent: Optional[ca.SX] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dims = ClsDims(self.dims)
        self.q = q
        self.v = v
        self.f_v = f_v
        self.f_c = f_c
        self.mu = mu
        self.e = e
        self.M = M
        self.inv_M = inv_M
        self.friction_exists = False
        self.J_normal = J_normal
        self.J_tangent = J_tangent
        self.D_tangent = D_tangent

        self.__backfill()

    def __backfill(self):
        dims = self.dims

    
        if dims.n_x % 2 != 0:
            raise RuntimeError(f"The state x of a Cls model must be (q,v) and therefore have an even number of entries, got {dims.n_x}.")
        dims.n_q = dims.n_x//2
        dims.n_v = dims.n_x//2

        if self.q is None:
            self.q = self.x[0:dims.n_q]
        if self.v is None:
            self.v = self.x[dims.n_q:]

        if self.f_v.size(1) != dims.n_v:
            raise RuntimeError(f"f_v has incorrect dimension, it must have the same dimension as v ({dims.n_v}), got {self.f_v.size(1)}.")

        dims.n_c = self.f_c.size(1)

       
        if self.mu is None:
            self.mu = np.zeros(dims.n_c)
        else:
            self.mu = self.__broadcast_to_contacts(self.mu, "mu")
            if np.any(self.mu < 0):
                raise RuntimeError("The coefficients of friction mu should be nonnegative.")
        self.friction_exists = bool(np.any(self.mu > 0))

        if self.e is None:
            raise RuntimeError("Please provide a coefficient of restitution via e.")
        self.e = self.__broadcast_to_contacts(self.e, "e")
        if np.any(self.e < 0) or np.any(self.e > 1):
            raise RuntimeError("The coefficient of restitution e should be in [0,1].")

        if self.M is None:
            self.M = np.eye(dims.n_q)
        elif np.any(np.array(self.M.shape) != dims.n_q):
            raise RuntimeError(f"Inertia matrix M must be {dims.n_q}x{dims.n_q}, got {self.M.shape[0]}x{self.M.shape[1]}.")
        if self.inv_M is None:
            if isinstance(self.M, np.ndarray):
                self.inv_M = np.linalg.inv(self.M)
            else:
                self.inv_M = ca.inv(self.M)

        
        if self.J_normal is None:
            self.J_normal = ca.jacobian(self.f_c, self.q).T
        elif self.J_normal.size(1) != dims.n_q or self.J_normal.size(2) != dims.n_c:
            raise RuntimeError(f"J_normal must be a {dims.n_q}x{dims.n_c} matrix, got {self.J_normal.size(1)}x{self.J_normal.size(2)}.")

        # TODO(@stefan) implement the Conic and Polyhedral friction cones. n_t and n_tangents are
        # already laid out the way the friction variables will need them, cf. the MATLAB
        # implementation in `+nosnoc/+model/Cls.m`.
        if self.friction_exists:
            raise NotImplementedError("Friction is not yet implemented for the Python CLS, please use mu=0 for all contacts.")
        dims.n_t = 0
        dims.n_tangents = 0

    def __broadcast_to_contacts(self, val, name: str) -> np.ndarray:
        """
        Take a scalar or vector coefficient and return a vector with one entry per contact.
        """
        if isinstance(val, Real):
            return float(val)*np.ones(self.dims.n_c)
        val = np.asarray(val, dtype=float).flatten()
        if val.shape[0] == 1:
            return val[0]*np.ones(self.dims.n_c)
        if val.shape[0] != self.dims.n_c:
            raise RuntimeError(f"The length of {name} has to be one or match the number of contacts ({self.dims.n_c}), got {val.shape[0]}.")
        return val
