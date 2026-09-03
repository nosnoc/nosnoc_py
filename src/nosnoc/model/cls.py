from .base import Base, BaseDims
from ..dims import Dims
from ..nosnoc_types import FrictionModel

from typing import Optional, List, Tuple
from numbers import Real
from warnings import warn

import casadi as ca
import numpy as np


class ClsDims(Dims):
    def __init__(self, parent: BaseDims):
        super().__init__(parent)
        self.n_q = 0 # Number of generalized coordinates.
        self.n_v = 0 # Number of generalized velocities, equal to n_q.
        self.n_c = 0 # Number of possible contacts.
        # Spatial dimension of a contact, 2 for a planar and 3 for a spatial problem. Derived from
        # the column count of J_tangent, stays 0 if only D_tangent was provided.
        self.n_dim_contact = 0
        # Tangential directions per contact used by the Conic friction model. This is the dimension
        # of the tangent space, n_dim_contact-1, i.e. geometry rather than a modelling choice: 1 for
        # a planar contact, 2 for a spatial one. The number of *cone constraints* is always n_c, one
        # exact Coulomb cone per contact.
        self.n_t_conic = 0
        # Polyhedral generators (facets) per contact, the column count of D_tangent per contact.
        # Unlike n_t_conic this *is* an approximation-fidelity choice: 2 in the plane (where the
        # polyhedral cone is exact) and 4 or more in space.
        self.n_facets = 0
        # NOTE: the resolved n_t / n_tangents depend on opts.friction_model and therefore live on
        # `dcs.ClsDcsDims`, not here. `Dims.__setattr__` writes through to the parent whenever the
        # attribute already exists there, so declaring them here would make every DCS built from
        # this model overwrite the dims of every other one.


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

    Coulomb friction is added by passing a nonzero coefficient of friction `mu` together with a
    tangent Jacobian. Which of the two friction models is used is an *option*
    (`opts.friction_model`), not a property of the model, so both tangent Jacobians are described
    here and the discretization picks the one it needs:

    * `FrictionModel.CONIC` uses `J_tangent`, whose columns span the tangent space at each contact
      (1 column per contact in the plane, 2 in space), and imposes the exact cone
      $\|\lambda_{\mathrm{t}}^i\|_2 \le \mu^i\lambda_{\mathrm{n}}^i$.
    * `FrictionModel.POLYHEDRAL` uses `D_tangent`, whose columns are the generators of a polyhedral
      approximation of that cone. In the plane the approximation is exact.

    If `D_tangent` is omitted it is built from `J_tangent` as `[t_1, -t_1, t_2, -t_2, ...]` per
    contact. Note the per-contact blocking: the columns belonging to contact $i$ are contiguous.
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
                 # Tangent basis, $n_q \times (n_t^{\mathrm{conic}} n_c)$, blocked per contact.
                 # Required for Conic friction, and used to build D_tangent if that is omitted.
                 # Its columns should be orthonormal within each contact block.
                 J_tangent: Optional[ca.SX] = None,
                 # Generators of the polyhedral friction cone, $n_q \times (n_{\mathrm{facets}} n_c)$,
                 # blocked per contact. Required for Polyhedral friction; built from J_tangent as
                 # [t_1, -t_1, t_2, -t_2] per contact when omitted. Within each contact block every
                 # column must have its negation present, and all columns should be unit vectors.
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

        if self.J_tangent is not None:
            self.J_tangent = ca.SX(self.J_tangent)
        if self.D_tangent is not None:
            self.D_tangent = ca.SX(self.D_tangent)
        if self.friction_exists:
            self.__setup_friction()

    def __setup_friction(self):
        """
        Resolve the tangent Jacobians and the friction dimensions that do not depend on options.

        `n_t_conic` and `n_facets` are both derived here; which of the two the discretization uses
        is decided later by `friction_dims`, so that one model can feed several discretizations with
        different `opts.friction_model`.
        """
        dims = self.dims

        if self.J_tangent is None and self.D_tangent is None:
            raise RuntimeError(
                "A model with friction (mu > 0) needs a tangent Jacobian: provide J_tangent "
                "(required by FrictionModel.CONIC, and used to build D_tangent automatically) "
                "and/or D_tangent (required by FrictionModel.POLYHEDRAL).")

        if self.J_tangent is not None:
            dims.n_t_conic = self.__tangent_cols_per_contact(self.J_tangent, "J_tangent")
            dims.n_dim_contact = dims.n_t_conic + 1

        if self.D_tangent is None:
            # Build the polyhedral generators from the tangent basis, blocked per contact as
            # [t_1, -t_1, t_2, -t_2, ...]. Deliberately *not* [J_tangent, -J_tangent], which puts
            # all the positive directions first and so mis-associates facets with contacts as soon
            # as there is more than one contact.
            cols = []
            for ii in range(dims.n_c):
                for kk in range(dims.n_t_conic):
                    t = self.J_tangent[:, ii*dims.n_t_conic + kk]
                    cols += [t, -t]
            self.D_tangent = ca.horzcat(*cols)
            dims.n_facets = 2*dims.n_t_conic
        else:
            dims.n_facets = self.__tangent_cols_per_contact(self.D_tangent, "D_tangent")
            if dims.n_facets % 2 != 0:
                raise RuntimeError(
                    f"D_tangent must have an even number of columns per contact so that the "
                    f"polyhedral cone is symmetric, got {dims.n_facets}.")
            self.__check_d_tangent_pairing()

        self.__check_tangent_conditioning()

    def __tangent_cols_per_contact(self, J, name: str) -> int:
        """Validate the shape of a tangent Jacobian and return its column count per contact."""
        dims = self.dims
        if J.size(1) != dims.n_q:
            raise RuntimeError(
                f"{name} must have one row per generalized coordinate ({dims.n_q}), got {J.size(1)}.")
        if J.size(2) % dims.n_c != 0:
            raise RuntimeError(
                f"{name} must have the same number of columns for each of the {dims.n_c} contacts, "
                f"blocked per contact, got {J.size(2)} columns in total.")
        return J.size(2)//dims.n_c

    def __eval_at_x0(self, expr, name: str) -> Optional[np.ndarray]:
        """
        Evaluate a (possibly q dependent) tangent Jacobian at x0, for the numerical sanity checks.

        Returns None if the expression depends on anything but the state, or is not finite there, in
        which case the caller skips its check rather than failing on a perfectly valid model.
        """
        try:
            val = np.array(ca.Function(name, [self.x], [expr])(self.x0))
        except Exception:
            return None
        return val if np.all(np.isfinite(val)) else None

    def __check_d_tangent_pairing(self):
        """
        Every generator of the polyhedral cone must have its negation among the generators of the
        *same* contact, otherwise the friction force cannot oppose an arbitrary sliding direction.
        This is the check that catches a D_tangent built as [J_tangent, -J_tangent], which is only
        correct for a single contact.
        """
        dims = self.dims
        D = self.__eval_at_x0(self.D_tangent, "D_tangent_at_x0")
        if D is None:
            return
        scale = max(np.max(np.abs(D)), 1.0)
        for ii in range(dims.n_c):
            lo, hi = ii*dims.n_facets, (ii+1)*dims.n_facets
            block = D[:, lo:hi]
            for jj in range(dims.n_facets):
                partner = np.min(np.linalg.norm(block + block[:, [jj]], axis=0))
                if partner > 1e-9*scale:
                    raise RuntimeError(
                        f"Column {lo+jj} of D_tangent has no matching -column within the block of "
                        f"contact {ii} (columns {lo}:{hi}). The polyhedral generators must be "
                        f"blocked per contact and symmetric, e.g. "
                        f"[t_1, -t_1, t_2, -t_2] for each contact in turn. Note that "
                        f"[J_tangent, -J_tangent] has the wrong blocking for more than one contact.")

    def __check_tangent_conditioning(self):
        """
        Warn about tangent Jacobians that are shaped right but silently change the friction law.

        Both friction models write their cone bound on the *coefficients* rather than on the force,
        so the columns carry an implicit normalization assumption: the conic bound
        ||lambda_t|| <= mu*lambda_n only means isotropic Coulomb friction if J_tangent has
        orthonormal columns, and the polyhedral budget sum(lambda_t) <= mu*lambda_n weights every
        generator equally, so the columns of D_tangent must be unit vectors. Violating either gives
        anisotropic friction with no error anywhere, so it is worth a warning.
        """
        dims = self.dims
        if self.J_tangent is not None:
            J = self.__eval_at_x0(self.J_tangent, "J_tangent_at_x0")
            if J is not None:
                for ii in range(dims.n_c):
                    block = J[:, ii*dims.n_t_conic:(ii+1)*dims.n_t_conic]
                    if np.linalg.norm(block.T@block - np.eye(dims.n_t_conic)) > 1e-6:
                        warn(f"The columns of J_tangent for contact {ii} are not orthonormal at x0. "
                             "The conic friction model bounds ||lambda_t||, which only equals the "
                             "magnitude of the tangential force J_tangent@lambda_t for an "
                             "orthonormal basis; otherwise the friction cone becomes an elliptic "
                             "cone and friction is anisotropic.", stacklevel=4)
        D = self.__eval_at_x0(self.D_tangent, "D_tangent_at_x0")
        if D is not None:
            norms = np.linalg.norm(D, axis=0)
            if np.any(np.abs(norms - 1.0) > 1e-6):
                warn("The columns of D_tangent are not unit vectors at x0. The polyhedral friction "
                     "model spends a single budget mu*lambda_n over all generators, so a longer "
                     "column reaches further than its neighbours and the friction cone is "
                     "anisotropic. Normalize the columns of D_tangent (or of J_tangent, from which "
                     "it is built).", stacklevel=4)

    def friction_dims(self, friction_model: FrictionModel) -> Tuple[int, int]:
        """
        Resolve `(n_t, n_tangents)` for a friction model.

        `n_t` is the number of tangential multipliers per contact and `n_tangents = n_t*n_c` the
        total. This is deliberately a query rather than an assignment: the answer depends on options
        the model does not own, and several discretizations may share one model.
        """
        dims = self.dims
        if not self.friction_exists:
            return 0, 0
        if friction_model == FrictionModel.CONIC:
            if self.J_tangent is None:
                raise RuntimeError(
                    "FrictionModel.CONIC needs the tangent basis J_tangent, but only D_tangent was "
                    "provided. Either pass J_tangent or use FrictionModel.POLYHEDRAL.")
            if dims.n_t_conic == 1:
                raise RuntimeError(
                    "FrictionModel.CONIC was selected for a planar contact (J_tangent has a single "
                    "column per contact, so the tangent space is one dimensional). In the plane the "
                    "polyhedral friction cone is exact and yields an LCP rather than an NCP, which "
                    "behaves much better numerically. Use FrictionModel.POLYHEDRAL.")
            n_t = dims.n_t_conic
        else:
            n_t = dims.n_facets
        return n_t, n_t*dims.n_c

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
