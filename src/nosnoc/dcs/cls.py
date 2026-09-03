from typing import override

from ..model import Cls as ClsModel, ClsDims
from ..dims import Dims
from ..nosnoc_types import FrictionModel, ConicModelSwitchHandling
from .base import Base

import casadi as ca


class ClsDcsDims(Dims):
    def __init__(self, parent: ClsDims):
        super().__init__(parent)
        self.n_lambda_normal = 0
        self.n_y_gap = 0
        # Resolved from opts.friction_model. These are safe to keep here rather than on the model:
        # `ClsDims` deliberately does not declare them, so `Dims.__setattr__` cannot write them
        # through to the model and two discretizations of one model stay independent.
        self.n_t = 0
        self.n_tangents = 0


class Cls(Base):
    r"""
    Reformulation of a Complementarity Lagrangian System into a DCS.

    The contact forces are determined by the complementarity conditions

        0 <= lambda_normal  _|_  y_gap >= 0,   y_gap = f_c(q),

    and, at the boundaries of the finite elements, the impulse equations determine either a state
    jump or the continuity of the velocities. `y_gap` and `Y_gap` are lifting variables for f_c(q),
    which keep the complementarity conditions linear in the variables.

    Coulomb friction adds a tangential force lambda_tangent, determined by the maximum dissipation
    principle. Two reformulations of that principle exist:

    * `POLYHEDRAL` writes the friction cone as the convex hull of the generators `D_tangent`, which
      turns the maximum dissipation principle into an LCP (Stewart-Trinkle). `lambda_tangent >= 0`
      holds the force magnitude along each generator, `gamma_d` is the sliding speed multiplier,
      `delta_d` the per generator relative velocity and `beta_d` the unused friction budget. The
      switch between sticking and sliding is the complementarity `beta_d _|_ gamma_d`.
    * `CONIC` keeps the exact cone ||lambda_t|| <= mu*lambda_n. `gamma` is the multiplier of the
      (squared) cone constraint and `beta` its lifted slack, complementary to each other. The
      `ConicModelSwitchHandling` variants decide whether sign changes of the tangential velocity get
      their own variables (`p_vt`/`n_vt`, plus `alpha_vt` for `LP`) so that FESD can isolate them.

    Which of the two is used is `opts.friction_model`, read once at construction: exactly one
    reformulation is built, and `dcs.g_alg`, `dcs.f_x_rk` and friends are its equations. Which
    variables that reformulation actually has varies with the options, so instead of hard coding
    the stacking order in each place that needs it, `_build_friction_variables` records it once in
    `z_alg_blocks` / `z_impulse_blocks` / `z_alg_f_x_blocks` and the discrete time problem and the
    integrator rebuild their own stacks from those names. That is what keeps the two sides from
    drifting apart as friction blocks appear and disappear.

    The apex regularization `eps_t` is not baked in: it is a symbolic parameter, threaded through
    the RK parameter vector next to `h_rescale` and populated by the discrete time problem, so it
    can be changed between solves without rebuilding the reformulation.
    """

    def __init__(self, model: ClsModel, opts):
        self.opts = opts
        self.dims = ClsDcsDims(model.dims)
        super().__init__(model)

    def _selected_friction_model(self):
        """
        The friction model this reformulation is built for, or None when there is no friction.

        Asking the model for the dimensions here is what surfaces the actionable errors (a planar
        contact with the conic cone, a conic cone without a tangent basis) while the dcs is being
        built, rather than as a shape mismatch further downstream.
        """
        if not self.model.friction_exists:
            return None
        friction_model = self.opts.friction_model
        self.model.friction_dims(friction_model)
        return friction_model

    # ------------------------------------------------------------------ variable generation

    @override
    def _generate_variables(self):
        """Generate the required variables for the dcs"""
        dims = self.dims
        dims.n_lambda_normal = dims.n_c
        dims.n_y_gap = dims.n_c

        self.lambda_normal = ca.SX.sym("lambda_normal", dims.n_c)
        self.y_gap = ca.SX.sym("y_gap", dims.n_c)

        self.Lambda_normal = ca.SX.sym("Lambda_normal", dims.n_c)
        self.Y_gap = ca.SX.sym("Y_gap", dims.n_c)

        # Positive and negative parts of the restitution law residual. They are used to encode the absolute value
        # in the aggregated impulse complementarity, cf. Eq. (A.2) of the FESD-J paper.
        self.P_vn = ca.SX.sym("P_vn", dims.n_c)
        self.N_vn = ca.SX.sym("N_vn", dims.n_c)

        # Symbolic apex regularization, populated by the discrete time problem.
        self.eps_t = ca.SX.sym("eps_t")

        self.friction_model = self._selected_friction_model()
        self.switch_handling = self.opts.conic_model_switch_handling
        self._build_friction_variables()

        self.z_all = ca.vertcat(self.z_alg, self.model.z)

    def _build_friction_variables(self):
        """
        Create the symbols and the stacking order of the selected friction reformulation.

        `syms` maps the name of every contact variable to its symbol, and the three `*_blocks`
        lists name the variables making up `z_alg`, `z_impulse` and `z_alg_f_x`, in order. Those
        names are the contract with the discrete time problem and the integrator, which rebuild
        their own stacks from them; see the class docstring.
        """
        model, dims = self.model, self.dims
        friction_model, switch_handling = self.friction_model, self.switch_handling

        # Human readable name of the reformulation that was built, for debugging.
        self.friction_variant = "frictionless" if friction_model is None else (
            friction_model.name if switch_handling is None
            else f"{friction_model.name}/{switch_handling.name}")
        dims.n_t, dims.n_tangents = (model.friction_dims(friction_model)
                                     if friction_model is not None else (0, 0))

        self.syms = {}

        def sym(nm, size):
            self.syms[nm] = ca.SX.sym(nm, size)
            return self.syms[nm]

        self.z_alg_blocks = ["lambda_normal", "y_gap"]
        self.z_impulse_blocks = ["Lambda_normal", "Y_gap", "P_vn", "N_vn"]
        self.z_alg_f_x_blocks = ["lambda_normal"]
        self.syms.update(lambda_normal=self.lambda_normal, y_gap=self.y_gap,
                         Lambda_normal=self.Lambda_normal, Y_gap=self.Y_gap,
                         P_vn=self.P_vn, N_vn=self.N_vn)

        if friction_model is not None:
            n_tangents = dims.n_tangents
            sym("lambda_tangent", n_tangents)
            sym("Lambda_tangent", n_tangents)
            self.z_alg_blocks.append("lambda_tangent")
            self.z_impulse_blocks.append("Lambda_tangent")
            self.z_alg_f_x_blocks.append("lambda_tangent")

            if friction_model == FrictionModel.POLYHEDRAL:
                for nm, size in (("gamma_d", dims.n_c), ("beta_d", dims.n_c),
                                 ("delta_d", n_tangents)):
                    sym(nm, size)
                for nm, size in (("Gamma_d", dims.n_c), ("Beta_d", dims.n_c),
                                 ("Delta_d", n_tangents)):
                    sym(nm, size)
                self.z_alg_blocks += ["gamma_d", "beta_d", "delta_d"]
                self.z_impulse_blocks += ["Gamma_d", "Beta_d", "Delta_d"]
            else:
                for nm in ("gamma", "beta", "Gamma", "Beta"):
                    sym(nm, dims.n_c)
                self.z_alg_blocks += ["gamma", "beta"]
                self.z_impulse_blocks += ["Gamma", "Beta"]
                if switch_handling != ConicModelSwitchHandling.PLAIN:
                    for nm in ("p_vt", "n_vt", "P_vt", "N_vt"):
                        sym(nm, n_tangents)
                    self.z_alg_blocks += ["p_vt", "n_vt"]
                    self.z_impulse_blocks += ["P_vt", "N_vt"]
                    if switch_handling == ConicModelSwitchHandling.LP:
                        sym("alpha_vt", n_tangents)
                        sym("Alpha_vt", n_tangents)
                        self.z_alg_blocks.append("alpha_vt")
                        self.z_impulse_blocks.append("Alpha_vt")

        self.z_alg = ca.vertcat(*[self.syms[b] for b in self.z_alg_blocks])
        self.z_impulse = ca.vertcat(*[self.syms[b] for b in self.z_impulse_blocks])
        self.z_alg_f_x = ca.vertcat(*[self.syms[b] for b in self.z_alg_f_x_blocks])

    # ------------------------------------------------------------------ equation generation

    def _friction_jacobian(self, friction_model):
        """The tangent Jacobian that multiplies lambda_tangent in the equations of motion."""
        if friction_model == FrictionModel.POLYHEDRAL:
            return self.model.D_tangent
        return self.model.J_tangent

    def _contact_force(self, friction_model, lambda_normal, lambda_tangent):
        """Generalized force produced by the contact multipliers."""
        f = self.model.J_normal@lambda_normal
        if friction_model is not None:
            f = f + self._friction_jacobian(friction_model)@lambda_tangent
        return f

    def _friction_equations(self, friction_model, switch_handling, v,
                            lambda_normal, lambda_tangent, aux):
        """
        Friction equations at a stage point or, with the impulse variables, at an impact.

        `aux` maps the role of each auxiliary variable to the symbol to use, so that the stage
        equations and their capitalized impulse twins are generated from one place. `v` is the
        velocity the friction acts on: the stage velocity, or the *post* impact velocity.
        """
        model = self.model
        dims = self.dims
        J_f = self._friction_jacobian(friction_model)
        g = []
        for ii in range(dims.n_c):
            lo, hi = ii*dims.n_t, (ii+1)*dims.n_t
            v_t = J_f[:, lo:hi].T@v
            if friction_model == FrictionModel.POLYHEDRAL:
                # Remaining friction budget, and the per generator relative velocity offset by the
                # sliding speed multiplier. beta_d _|_ gamma_d is the stick/slip switch and
                # lambda_tangent _|_ delta_d picks the generator opposing the sliding direction.
                g.append(aux["beta"][ii]
                         - (model.mu[ii]*lambda_normal[ii] - ca.sum1(lambda_tangent[lo:hi])))
                g.append(aux["delta"][lo:hi] - (v_t + aux["gamma"][ii]))
            else:
                # Stationarity of the maximum dissipation principle for the squared cone
                # constraint, and the lifted cone slack. The eps_t shift opens up the apex of the
                # cone, where the gradient of the unregularized constraint vanishes and LICQ fails.
                #
                # The velocity term is scaled by the cone radius mu*lambda_n. Wherever the contact
                # carries force this is an identity: dividing by mu*lambda_n > 0 recovers the
                # textbook condition v_t + 2*gamma*lambda_t = 0 with gamma rescaled. It matters
                # when the contact is *open*: there the cone collapses to {0}, so lambda_t = 0 and
                # the unscaled equation would demand v_t = 0, which a body flying with tangential
                # motion cannot satisfy. The relaxed MPCC escapes that only by driving gamma to
                # infinity (we measured ~1e8), which wrecks the conditioning of the whole KKT
                # system and stalls the homotopy. With the scaling both sides vanish for an open
                # contact, gamma stays bounded at ~|v_t|/2, and the friction subproblem is as well
                # scaled as the polyhedral one. This is what the (declared but never used) MATLAB
                # option `kappa_friction_reg` was meant to address.
                cone_radius = model.mu[ii]*lambda_normal[ii]
                g.append(-cone_radius*v_t - 2*aux["gamma"][ii]*lambda_tangent[lo:hi])
                g.append(aux["beta"][ii] - ((model.mu[ii]*lambda_normal[ii])**2
                                            - ca.sumsqr(lambda_tangent[lo:hi] + self.eps_t)))
                if switch_handling != ConicModelSwitchHandling.PLAIN:
                    # Split the tangential velocity so that FESD can isolate its sign changes.
                    g.append(v_t - (aux["p_vt"][lo:hi] - aux["n_vt"][lo:hi]))
        return g

    def _friction_aux(self, friction_model, switch_handling, impulse: bool):
        """Map the auxiliary roles to the stage or the impulse symbols of the reformulation."""
        s = self.syms
        if friction_model == FrictionModel.POLYHEDRAL:
            if impulse:
                return {"beta": s["Beta_d"], "gamma": s["Gamma_d"], "delta": s["Delta_d"]}
            return {"beta": s["beta_d"], "gamma": s["gamma_d"], "delta": s["delta_d"]}
        aux = ({"beta": s["Beta"], "gamma": s["Gamma"]} if impulse
               else {"beta": s["beta"], "gamma": s["gamma"]})
        if switch_handling != ConicModelSwitchHandling.PLAIN:
            aux.update({"p_vt": s["P_vt"], "n_vt": s["N_vt"]} if impulse
                       else {"p_vt": s["p_vt"], "n_vt": s["n_vt"]})
        return aux

    @override
    def _generate_expressions(self):
        """Generate the required equations and functions for the dcs"""
        model = self.model
        dims = self.dims
        J_n = model.J_normal

        # Functions shared by every friction reformulation.
        self.f_q_fun = ca.Function('f_q', [model.x, model.z, model.u, model.v_global, model.p], [model.f_q])
        self.g_z_fun = ca.Function('g_z', [model.x, model.z, model.u, model.v_global, model.p], [model.g_z])
        self.M_fun = ca.Function('M_fun', [model.x], [model.M])
        self.invM_fun = ca.Function('invM_fun', [model.x], [model.inv_M])
        self.f_c_fun = ca.Function('f_c_fun', [model.x], [model.f_c])
        self.J_normal_fun = ca.Function('J_normal_fun', [model.x], [J_n])
        if model.friction_exists:
            if model.J_tangent is not None:
                self.J_tangent_fun = ca.Function('J_tangent_fun', [model.x], [model.J_tangent])
            self.D_tangent_fun = ca.Function('D_tangent_fun', [model.x], [model.D_tangent])
        self.g_path_fun = ca.Function('g_path', [model.x, model.z, model.u, model.v_global, model.p], [model.g_path])
        self.G_path_fun = ca.Function('G_path', [model.x, model.z, model.u, model.v_global, model.p], [model.G_path])
        self.H_path_fun = ca.Function('H_path', [model.x, model.z, model.u, model.v_global, model.p], [model.H_path])
        self.g_terminal_fun = ca.Function('g_terminal', [model.x, model.z, model.v_global, model.p_global], [model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [model.x, model.z, model.v_global, model.p], [model.f_q_T])

        # The RK functions additionally take h_rescale as a parameter. lambda_normal is a contact
        # force in FESD (h_rescale = 1) but a contact impulse in the non-FESD implicit-Euler scheme,
        # where the ODE right hand side divides it by the fixed step length. Following the MATLAB
        # Cls.m, the division lives inside f_x only; contact force is not rescaled in the quadrature and algebraic equations.
        self.h_rescale = ca.SX.sym("h_rescale")

        friction_model, switch_handling = self.friction_model, self.switch_handling
        lam_t = self.syms.get("lambda_tangent")
        Lam_t = self.syms.get("Lambda_tangent")

        self.f_x = ca.vertcat(
            model.v,
            model.inv_M@(model.f_v + self._contact_force(friction_model,
                                                         self.lambda_normal, lam_t)))

        g_alg = [self.y_gap - model.f_c]
        if friction_model is not None:
            g_alg += self._friction_equations(
                friction_model, switch_handling, model.v, self.lambda_normal, lam_t,
                self._friction_aux(friction_model, switch_handling, impulse=False))
        self.g_alg = ca.vertcat(*g_alg)

        v_post_impact = ca.SX.sym("v_post_impact", dims.n_q)
        v_pre_impact = ca.SX.sym("v_pre_impact", dims.n_q)

        g_impulse = [model.M@(v_post_impact - v_pre_impact)
                     - self._contact_force(friction_model, self.Lambda_normal, Lam_t)]
        g_impulse.append(self.Y_gap - model.f_c)
        for ii in range(dims.n_c):
            g_impulse.append(
                self.P_vn[ii] - self.N_vn[ii]
                - model.J_normal[:,ii].T@(v_post_impact + model.e[ii]*v_pre_impact)
            )
        if friction_model is not None:
            # The friction impulse acts on the post impact velocity.
            g_impulse += self._friction_equations(
                friction_model, switch_handling, v_post_impact, self.Lambda_normal, Lam_t,
                self._friction_aux(friction_model, switch_handling, impulse=True))
        self.g_impulse = ca.vertcat(*g_impulse)

        self.f_x_fun = ca.Function(
            'f_x', [model.x, model.z, self.z_alg_f_x, model.u, model.v_global, model.p, self.eps_t],
            [self.f_x, model.f_q])
        self.g_alg_fun = ca.Function(
            'g_alg', [model.x, model.z, self.z_alg, model.v_global, model.p, self.eps_t],
            [self.g_alg])
        self.g_impulse_fun = ca.Function(
            'g_impulse',
            [model.q, v_post_impact, v_pre_impact, self.z_impulse, model.v_global, model.p,
             self.eps_t],
            [self.g_impulse])

        f_x_rk_expr = ca.vertcat(
            model.v,
            model.inv_M@(model.f_v + self._contact_force(
                friction_model, self.lambda_normal, lam_t)/self.h_rescale))
        p_rk = ca.vertcat(model.u, model.v_global, model.p, self.h_rescale, self.eps_t)
        z_rk = ca.vertcat(model.x, model.z, self.z_alg)

        self.f_x_rk = ca.Function('f_x_rk', [z_rk, p_rk], [f_x_rk_expr])
        self.f_q_rk = ca.Function('f_q_rk', [z_rk, p_rk], [model.f_q])
        self.g_rk = ca.Function('g_rk', [z_rk, p_rk], [ca.vertcat(model.g_z, self.g_alg)])
