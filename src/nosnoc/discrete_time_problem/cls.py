from typing import override
from warnings import warn

import casadi as ca
import numpy as np

from .base import Base
from vdx.vartypes import *

from ..nosnoc_types import (RKRepresentation, CrossComplementarityMode, StepEquilibrationMode,
                            ClsDiscretization, RKScheme, FrictionModel, ConicModelSwitchHandling)


class Cls(Base):
    r"""
    Discrete time problem (MPCC) for a Complementarity Lagrangian System.

    Two discretizations of the impact are supported, selected by ``opts.cls_discretization``:
    FESD-J (impulse + velocity jump at the finite element boundaries, exact) and Patel's relaxed
    orthogonal collocation (velocity continuity + contact force over a shrinking element,
    approximate).
    """

    def __init__(self, dcs, opts):
        self.__apply_time_stepping_defaults(opts)
        self.__check_restitution_supported(dcs.model, opts)
        # Friction dependent symbols, equations and functions, grouped by the dcs.
        self.variant = dcs.variant
        super().__init__(dcs, opts)

    def __apply_time_stepping_defaults(self, opts):
        """
        Without FESD a CLS is discretized with the implicit Euler time-stepping scheme, which is
        Radau IIA with a single stage.
        """
        if opts.use_fesd:
            return
        if opts.n_s != 1 or opts.rk_scheme != RKScheme.RADAU_IIA:
            warn("use_fesd = 0 with a CLS implies using the implicit Euler time-stepping scheme, "
                 "setting n_s = 1, rk_scheme = RADAU_IIA.",
                 stacklevel=3) # points at the caller that built the discrete time problem
        opts.rk_scheme = RKScheme.RADAU_IIA
        opts.n_s = 1

    def __check_restitution_supported(self, model, opts):
        """
        Reject a nonzero coefficient of restitution in the discretizations that cannot represent it.

        Newton's restitution law enters the problem only through the impulse equations
        `dcs.g_impulse`, whose term $J_n^\\top(v(t_s^+) + e\\,v(t_s^-))$ carries `e`. Those equations
        are generated exactly when FESD-J is active, see `_generate_direct_transcription_constraints`;
        otherwise the finite element boundary gets `v_continuity` instead, the velocity is continuous
        and every impact comes out plastic. A nonzero `e` would then be silently ignored rather than
        approximated, so it is refused here.

        """
        if not np.any(np.asarray(model.e) > 0.0):
            return
        if opts.cls_discretization == ClsDiscretization.RELAXED_OC:
            raise RuntimeError(
                "cls_discretization = RELAXED_OC has no impulse variables. The velocity is "
                "continuous across a finite element boundary, so every impact is plastic and the "
                f"coefficient of restitution e = {model.e} cannot be represented. Use "
                "cls_discretization = ClsDiscretization.FESD_J, or set e = 0.")
        if not opts.use_fesd:
            raise RuntimeError(
                "use_fesd = False selects the implicit Euler time stepping scheme, which has no "
                "impulse variables and therefore produces plastic impacts only. The coefficient "
                f"of restitution e = {model.e} cannot be represented. Use use_fesd = True "
                "together with cls_discretization = ClsDiscretization.FESD_J, or set e = 0.")

    def _is_relaxed_oc(self):
        """True if the relaxed orthogonal-collocation formulation PATEL is selected."""
        return self.opts.cls_discretization == ClsDiscretization.RELAXED_OC

    def _h_rescale(self, ii):
        """
        Scale that converts the stage contact multiplier to the correct units in the ODE RHS.

        In FESD-J ``lambda_normal`` is a genuine contact force so no rescaling is applied. In
        non-FESD-J implicit-Euler time-stepping it is instead a contact impulse over the fixed step,
        so ``f_x`` divides it by the step length ``h_0``; the ``h *`` from the Euler integration then
        cancels the division and the impulse acts directly on the velocity.
        """
        if self.opts.use_fesd:
            return 1.0
        return self.opts.h_k[ii-1] / self.opts.N_finite_elements[ii-1]

    def _start_fe(self):
        """First finite element that may contain an impact (2 if initial impacts are excluded)."""
        return 2 if self.opts.no_initial_impacts else 1

    @override
    def _create_variables(self):
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp
        start_fe = self._start_fe()

        self._create_global_variables()
        self._create_initial_variables()
        self._create_speed_of_time_variables()
        self._create_u()

        for ii in range(1, opts.N_stages+1):
            self._create_h(ii)
            self._create_xvz_cls(ii)

            # Contact force and lifted gap function at the RK stage points. y_gap is additionally
            # defined at the right boundary point (index n_s+rbp), lambda_normal is not.
            self.w.lambda_normal[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+1)] = Primal(
                "lambda_normal", dims.n_c, lb=0.0, ub=opts.ub_lambda_normal, init=opts.initial_lambda_normal)
            self.w.y_gap[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal(
                "y_gap", dims.n_c, lb=0.0, ub=opts.ub_y_gap, init=opts.initial_y_gap)

        
            if self.model.friction_exists:
                self._create_stage_friction_variables(ii)

            if opts.use_fesd and not self._is_relaxed_oc():
                fe_range = range(start_fe, opts.N_finite_elements[ii-1]+1)
                self.w.Lambda_normal[ii,fe_range] = Primal("Lambda_normal", dims.n_c, lb=0.0, ub=opts.ub_Lambda_normal, init=opts.initial_Lambda_normal)
                self.w.P_vn[ii,fe_range] = Primal("P_vn", dims.n_c, lb=0.0, ub=opts.ub_P_vn, init=opts.initial_P_vn)
                self.w.N_vn[ii,fe_range] = Primal("N_vn", dims.n_c, lb=0.0, ub=opts.ub_N_vn, init=opts.initial_N_vn)
                self.w.Y_gap[ii,fe_range] = Primal("Y_gap", dims.n_c, lb=0.0, ub=opts.ub_Y_gap, init=opts.initial_Y_gap)
                if self.model.friction_exists:
                    self._create_impulse_friction_variables(ii, fe_range)

        self._handle_x_box_constraints()

    # ------------------------------------------------------ friction variable creation

    def _is_polyhedral(self):
        return self.opts.friction_model == FrictionModel.POLYHEDRAL

    def _switch_handling(self):
        return self.opts.conic_model_switch_handling

    def _create_stage_friction_variables(self, ii):
        """
        Friction multipliers at the RK stage points, laid out like `lambda_normal`.

        The tangential force is nonnegative per generator in the polyhedral model, where it is a
        magnitude along a fixed direction, but free in sign in the conic model, where it is a
        coordinate in the tangent basis.
        """
        opts = self.opts
        dims = self.dcs.dims
        fe = range(1, opts.N_finite_elements[ii-1]+1)
        stg = range(1, opts.n_s+1)

        if self._is_polyhedral():
            self.w.lambda_tangent[ii,fe,stg] = Primal(
                "lambda_tangent", self.variant.n_tangents, lb=0.0,
                ub=opts.ub_lambda_tangent, init=opts.initial_lambda_tangent)
            self.w.gamma_d[ii,fe,stg] = Primal(
                "gamma_d", dims.n_c, lb=0.0, ub=opts.ub_gamma_d, init=opts.initial_gamma_d)
            self.w.beta_d[ii,fe,stg] = Primal(
                "beta_d", dims.n_c, lb=0.0, ub=opts.ub_beta_d, init=opts.initial_beta_d)
            self.w.delta_d[ii,fe,stg] = Primal(
                "delta_d", self.variant.n_tangents, lb=0.0, ub=opts.ub_delta_d, init=opts.initial_delta_d)
            return

        self.w.lambda_tangent[ii,fe,stg] = Primal(
            "lambda_tangent", self.variant.n_tangents, lb=-opts.ub_lambda_tangent,
            ub=opts.ub_lambda_tangent, init=opts.initial_lambda_tangent)
        self.w.gamma[ii,fe,stg] = Primal(
            "gamma", dims.n_c, lb=0.0, ub=opts.ub_gamma, init=opts.initial_gamma)
        self.w.beta[ii,fe,stg] = Primal(
            "beta", dims.n_c, lb=0.0, ub=opts.ub_beta, init=opts.initial_beta)
        if self._switch_handling() != ConicModelSwitchHandling.PLAIN:
            self.w.p_vt[ii,fe,stg] = Primal(
                "p_vt", self.variant.n_tangents, lb=0.0, ub=opts.ub_p_vt, init=opts.initial_p_vt)
            self.w.n_vt[ii,fe,stg] = Primal(
                "n_vt", self.variant.n_tangents, lb=0.0, ub=opts.ub_n_vt, init=opts.initial_n_vt)
            if self._switch_handling() == ConicModelSwitchHandling.LP:
                # alpha_vt is a step function, so it is bounded to [0,1] and complementary to both
                # the positive and the negative part of the tangential velocity.
                self.w.alpha_vt[ii,fe,stg] = Primal(
                    "alpha_vt", self.variant.n_tangents, lb=0.0, ub=1.0, init=opts.initial_alpha_vt)

    def _create_impulse_friction_variables(self, ii, fe_range):
        """Friction impulses at the finite element boundaries, laid out like `Lambda_normal`."""
        opts = self.opts
        dims = self.dcs.dims

        if self._is_polyhedral():
            self.w.Lambda_tangent[ii,fe_range] = Primal(
                "Lambda_tangent", self.variant.n_tangents, lb=0.0,
                ub=opts.ub_Lambda_tangent, init=opts.initial_Lambda_tangent)
            self.w.Gamma_d[ii,fe_range] = Primal(
                "Gamma_d", dims.n_c, lb=0.0, ub=opts.ub_Gamma_d, init=opts.initial_Gamma_d)
            self.w.Beta_d[ii,fe_range] = Primal(
                "Beta_d", dims.n_c, lb=0.0, ub=opts.ub_Beta_d, init=opts.initial_Beta_d)
            self.w.Delta_d[ii,fe_range] = Primal(
                "Delta_d", self.variant.n_tangents, lb=0.0, ub=opts.ub_Delta_d, init=opts.initial_Delta_d)
            return

        self.w.Lambda_tangent[ii,fe_range] = Primal(
            "Lambda_tangent", self.variant.n_tangents, lb=-opts.ub_Lambda_tangent,
            ub=opts.ub_Lambda_tangent, init=opts.initial_Lambda_tangent)
        self.w.Gamma[ii,fe_range] = Primal(
            "Gamma", dims.n_c, lb=0.0, ub=opts.ub_Gamma, init=opts.initial_Gamma)
        self.w.Beta[ii,fe_range] = Primal(
            "Beta", dims.n_c, lb=0.0, ub=opts.ub_Beta, init=opts.initial_Beta)
        if self._switch_handling() != ConicModelSwitchHandling.PLAIN:
            self.w.P_vt[ii,fe_range] = Primal(
                "P_vt", self.variant.n_tangents, lb=0.0, ub=opts.ub_P_vt, init=opts.initial_P_vt)
            self.w.N_vt[ii,fe_range] = Primal(
                "N_vt", self.variant.n_tangents, lb=0.0, ub=opts.ub_N_vt, init=opts.initial_N_vt)
            if self._switch_handling() == ConicModelSwitchHandling.LP:
                self.w.Alpha_vt[ii,fe_range] = Primal(
                    "Alpha_vt", self.variant.n_tangents, lb=0.0, ub=1.0, init=opts.initial_Alpha_vt)

    def _create_xvz_cls(self, ii):
        """
        Like `Base._create_xvz`, but the differential state additionally lives at the left boundary
        point (third index 0), which the FESD-J impulse equations need. 
        """
        opts = self.opts
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp
        if opts.rk_representation in (RKRepresentation.INTEGRAL, RKRepresentation.DIFFERENTIAL_LIFT_X):
            self.w.x[ii,range(1, opts.N_finite_elements[ii-1]+1),range(opts.n_s+rbp+1)] = Primal("x", dims.n_x,
                                                                                                      lb=model.lbx, 
                                                                                                      ub=model.ubx, 
                                                                                                      init=model.x0)
        else:
            self.w.x[ii,range(1, opts.N_finite_elements[ii-1]+1),0] = Primal("x", dims.n_x, 
                                                                             lb=model.lbx, 
                                                                             ub=model.ubx, 
                                                                             init=model.x0)
            self.w.x[ii,range(1, opts.N_finite_elements[ii-1]+1), opts.n_s+rbp] = Primal("x", dims.n_x, 
                                                                                         lb=model.lbx, 
                                                                                         ub=model.ubx,
                                                                                         init=model.x0)

        if opts.rk_representation in (RKRepresentation.DIFFERENTIAL, RKRepresentation.DIFFERENTIAL_LIFT_X):
            self.w.v[ii,range(1, opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+1)] = Primal("v", dims.n_x)

        self.w.z[ii,range(1, opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal("z", dims.n_z, 
                                                                                               lb=model.lbz, 
                                                                                               ub=model.ubz, 
                                                                                               init=model.z0)

    # ------------------------------------------------------------------ helpers

    def _build_z_impulse(self, ii, jj):
        """Stacked impulse algebraics for `g_impulse_fun`, matching the dcs `z_impulse` order."""
        return ca.vertcat(*[getattr(self.w, name)[ii,jj]
                            for name in self.variant.z_impulse_blocks])

    @override
    def _build_prk(self, ii, jj):
        # The CLS RK functions take two extra parameters. h_rescale is the fixed step length in
        # the non-FESD implicit-Euler scheme where lambda_normal is an impulse rather than a force;
        # only f_x uses it, f_q_rk / g_rk ignore it. eps_t is the conic apex regularization, left
        # symbolic by the dcs so that it can be changed without rebuilding the reformulation.
        return ca.vertcat(
            self.w.u[ii],
            self.w.v_global[()],
            self._get_stage_parameters(ii),
            self._h_rescale(ii),
            self.opts.eps_t,
        )

    @override
    def _get_rk_stage_z(self, ii, jj, kk):
        # The stacked algebraic order must match the dcs `z_alg` used to build f_x_rk / f_q_rk /
        # g_rk, so it is rebuilt from the same block list rather than spelled out again here: the
        # friction blocks come and go with opts.friction_model and opts.conic_model_switch_handling,
        # and a mismatch would silently pair the wrong variable with the wrong equation. The
        # h_rescale of the contact multipliers happens inside f_x (via the prk parameter), so they
        # are stacked as they are.
        z_alg = [getattr(self.w, name)[ii,jj,kk] for name in self.variant.z_alg_blocks]
        if self.opts.rk_representation == RKRepresentation.INTEGRAL:
            head = [self.w.x[ii,jj,kk], self.w.z[ii,jj,kk]]
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL:
            head = [self.w.v[ii,jj,kk], self.w.z[ii,jj,kk]]
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL_LIFT_X:
            head = [self.w.v[ii,jj,kk], self.w.x[ii,jj,kk], self.w.z[ii,jj,kk]]
        return ca.vertcat(*head, *z_alg)


    @override
    def _generate_direct_transcription_constraints(self):
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp

        x_0 = self.w.x[0,0,opts.n_s].sym
        z_0 = self.w.z[0,0,opts.n_s].sym
        self.g.algebraic[0,0,opts.n_s] = Constraint(
            dcs.g_z_fun(x_0, z_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1)))

        x_prev = x_0  # last point of previous FE, needed for continuity conditions
        for ii in range(1, opts.N_stages+1):
            s_sot = self._get_stage_sot(ii)
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                h = self._get_fe_h(ii, jj)
                u_i = self.w.u[ii]
                v_global = self.w.v_global[()]
                p = self._get_stage_parameters(ii)
                
                q_prev = x_prev[0:dims.n_q]
                v_prev = x_prev[dims.n_q:]
                x_lbp = self.w.x[ii,jj,0].sym
                q_lbp = x_lbp[0:dims.n_q]
                v_lbp = x_lbp[dims.n_q:]
                self.g.q_continuity[ii,jj] = Constraint(q_prev - q_lbp)

            
                if opts.use_fesd and not self._is_relaxed_oc() and (jj != 1 or not opts.no_initial_impacts):
                    self.g.impulse[ii,jj] = Constraint(
                        self.variant.g_impulse_fun(q_lbp, v_lbp, v_prev, self._build_z_impulse(ii,jj),
                                                   v_global, p, opts.eps_t))
                   
                    if opts.eps_cls > 0:
                        step = opts.eps_cls if opts.fixed_eps_cls else h*opts.eps_cls
                        x_eps = ca.vertcat(q_lbp + step*v_lbp, v_lbp)
                        self.g.f_c_eps[ii,jj] = Constraint(dcs.f_c_fun(x_eps), lb=0.0, ub=np.inf)
                else:
                    self.g.v_continuity[ii,jj] = Constraint(v_lbp - v_prev)

                #  RK collocation, started from the (post impact) left boundary point
                z_ii_jj = self._build_z(ii, jj)
                prk_ii_jj = self._build_prk(ii, jj)
                x_end, q_end, dynamic, algebraic = self.rk.collocation_constraints(
                    x_lbp, z_ii_jj, prk_ii_jj, h, self.variant.f_x_rk, self.variant.f_q_rk,
                        self.variant.g_rk, sot=s_sot)
                for kk in range(1, opts.n_s+1):
                    self.g.dynamic[ii,jj,kk] = Constraint(dynamic[kk-1])
                    self.g.algebraic[ii,jj,kk] = Constraint(algebraic[kk-1])
                    self._rk_stage_path_constraints(ii, jj, kk)
                self.f += q_end

                x_ii_jj_end = self._get_x_end(ii, jj)
                if not self.rk.is_right_boundary_explicit():
                    self.g.dynamic[ii,jj,opts.n_s+1] = Constraint(x_end - x_ii_jj_end)
                    self.g.algebraic[ii,jj,opts.n_s+1] = Constraint(
                        dcs.g_z_fun(x_ii_jj_end , self.w.z[ii,jj,opts.n_s+1].sym, u_i, v_global, p))
                    # y_gap at the right boundary point is defined by the gap function there.
                    self.g.y_gap_rbp[ii,jj] = Constraint(
                        self.w.y_gap[ii,jj,opts.n_s+1].sym - dcs.f_c_fun(x_ii_jj_end))
                self._fe_path_constraints(ii, jj)
                x_prev = x_ii_jj_end 
            self._numerical_time_constraints(ii)
            self._stage_path_constraints(ii)

        self._terminal_constraint()
        self._terminal_objective()
        self._terminal_numerical_time_constraints()


    @override
    def _generate_complementarity_constraints(self):
        opts = self.opts

        if opts.use_fesd:
            
            if not self._is_relaxed_oc():
                self.__impulse_comp()
            if opts.cross_comp_mode == CrossComplementarityMode.STAGE_STAGE:
                self.__stage_stage()
            elif opts.cross_comp_mode == CrossComplementarityMode.FE_STAGE:
                self.__fe_stage()
            elif opts.cross_comp_mode == CrossComplementarityMode.STAGE_FE:
                self.__stage_fe()
            elif opts.cross_comp_mode == CrossComplementarityMode.FE_FE:
                self.__fe_fe()
        else:
            self.__standard()

    def __impulse_comp(self):
        """Aggregated impulse complementarities at the finite element boundaries."""
        opts = self.opts
        start_fe = self._start_fe()
        for ii in range(1, opts.N_stages+1):
            for jj in range(start_fe, opts.N_finite_elements[ii-1]+1):
                Gij = [self.w.Lambda_normal[ii,jj], self.w.P_vn[ii,jj]]
                Hij = [self.w.Y_gap[ii,jj] + self.w.P_vn[ii,jj] + self.w.N_vn[ii,jj],
                       self.w.N_vn[ii,jj]]
                G_f, H_f = self._friction_impulse_pairs(ii, jj)
                self.G.impulse_comp[ii,jj] = CConstraint(ca.vertcat(*Gij, *G_f))
                self.H.impulse_comp[ii,jj] = CConstraint(ca.vertcat(*Hij, *H_f))

    # ------------------------------------------------------ friction complementarity pairs

    def _friction_impulse_pairs(self, ii, jj):
        """
        Friction part of the aggregated impulse complementarity, between impulse variables only.
        """
        if not self.model.friction_exists:
            return [], []
        w = self.w
        if self._is_polyhedral():
            return ([w.Delta_d[ii,jj], w.Gamma_d[ii,jj]],
                    [w.Lambda_tangent[ii,jj], w.Beta_d[ii,jj]])
        G = [w.Gamma[ii,jj]]
        H = [w.Beta[ii,jj]]
        sh = self._switch_handling()
        if sh == ConicModelSwitchHandling.ABS:
            G.append(w.P_vt[ii,jj]);   H.append(w.N_vt[ii,jj])
        elif sh == ConicModelSwitchHandling.LP:
            G.append(w.Alpha_vt[ii,jj]);     H.append(w.P_vt[ii,jj])
            G.append(1 - w.Alpha_vt[ii,jj]); H.append(w.N_vt[ii,jj])
        return G, H

    def _friction_pairs_stage_impulse(self, ii, jj, stage_at, n_terms):
        """
        Friction pairs coupling stage quantities of finite element `jj` to its impulse quantities.

        `stage_at(name)` returns the stage side expression, either a single RK stage point or a sum
        over them, and `n_terms` is how many stage points that expression aggregates. The count is
        needed for the `LP` step function, whose complement over a sum of `n` stages is
        `n - sum(alpha_vt)` rather than `1 - alpha_vt`.
        """
        if not self.model.friction_exists:
            return [], []
        w = self.w
        if self._is_polyhedral():
            return ([stage_at("lambda_tangent"), stage_at("beta_d")],
                    [w.Delta_d[ii,jj], w.Gamma_d[ii,jj]])
        G = [stage_at("beta")]
        H = [w.Gamma[ii,jj]]
        sh = self._switch_handling()
        if sh == ConicModelSwitchHandling.ABS:
            # Both orientations, so that a sign change of the tangential velocity across the impact
            # is detected regardless of which side it happens on.
            G.append(stage_at("p_vt")); H.append(w.N_vt[ii,jj])
            G.append(w.P_vt[ii,jj]);    H.append(stage_at("n_vt"))
        elif sh == ConicModelSwitchHandling.LP:
            alpha = stage_at("alpha_vt")
            G.append(alpha);             H.append(w.P_vt[ii,jj])
            G.append(n_terms - alpha);   H.append(w.N_vt[ii,jj])
        return G, H

    def _friction_pairs_stage_stage(self, ii, jj, lhs_at, n_lhs, rhs_at):
        """Friction pairs between stage quantities of the same finite element."""
        if not self.model.friction_exists:
            return [], []
        if self._is_polyhedral():
            return ([lhs_at("lambda_tangent"), lhs_at("beta_d")],
                    [rhs_at("delta_d"), rhs_at("gamma_d")])
        G = [lhs_at("beta")]
        H = [rhs_at("gamma")]
        sh = self._switch_handling()
        if sh == ConicModelSwitchHandling.ABS:
            G.append(lhs_at("p_vt")); H.append(rhs_at("n_vt"))
        elif sh == ConicModelSwitchHandling.LP:
            alpha = lhs_at("alpha_vt")
            G.append(alpha);           H.append(rhs_at("p_vt"))
            G.append(n_lhs - alpha);   H.append(rhs_at("n_vt"))
        return G, H

    def _at_stage(self, ii, jj, kk):
        """Accessor for a friction variable at a single RK stage point."""
        return lambda name: getattr(self.w, name)[ii,jj,kk]

    def _sum_stages(self, ii, jj):
        """Accessor summing a friction variable over all RK stage points of a finite element."""
        return lambda name: ca.sum2(getattr(self.w, name)[ii,jj,:].sym)

    def __stage_stage(self):
        opts = self.opts
        rbp = self.rbp
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                Gij = []
                Hij = []
                use_impulse = (jj != 1 or not opts.no_initial_impacts) and not self._is_relaxed_oc()
                if use_impulse:
                    for kk in range(1, opts.n_s+1):
                        Gij.append(self.w.lambda_normal[ii,jj,kk])
                        Hij.append(self.w.Y_gap[ii,jj])
                        G_f, H_f = self._friction_pairs_stage_impulse(
                            ii, jj, self._at_stage(ii,jj,kk), 1)
                        Gij += G_f; Hij += H_f
                for kk in range(1, opts.n_s+1):
                    for rr in range(1, opts.n_s+rbp+1):
                        Gij.append(self.w.lambda_normal[ii,jj,kk])
                        Hij.append(self.w.y_gap[ii,jj,rr])
                    for rr in range(1, opts.n_s+1):
                        # The friction variables only live at the RK stage points, not at the right
                        # boundary point that y_gap additionally occupies.
                        G_f, H_f = self._friction_pairs_stage_stage(
                            ii, jj, self._at_stage(ii,jj,kk), 1, self._at_stage(ii,jj,rr))
                        Gij += G_f; Hij += H_f
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __fe_stage(self):
        opts = self.opts
        rbp = self.rbp
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lambda = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)
                sum_at = self._sum_stages(ii, jj)
                Gij = []
                Hij = []
                if (jj != 1 or not opts.no_initial_impacts) and not self._is_relaxed_oc():
                    Gij.append(sum_lambda)
                    Hij.append(self.w.Y_gap[ii,jj])
                    G_f, H_f = self._friction_pairs_stage_impulse(ii, jj, sum_at, opts.n_s)
                    Gij += G_f; Hij += H_f
                for rr in range(1, opts.n_s+rbp+1):
                    Gij.append(sum_lambda)
                    Hij.append(self.w.y_gap[ii,jj,rr])
                for rr in range(1, opts.n_s+1):
                    G_f, H_f = self._friction_pairs_stage_stage(
                        ii, jj, sum_at, opts.n_s, self._at_stage(ii,jj,rr))
                    Gij += G_f; Hij += H_f
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __stage_fe(self):
        opts = self.opts
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                use_Y = (jj != 1 or not opts.no_initial_impacts) and not self._is_relaxed_oc()
                y_gap_lb = self.w.Y_gap[ii,jj] if use_Y else 0
                sum_y_gap = y_gap_lb + ca.sum2(self.w.y_gap[ii,jj,:].sym)
                sum_at = self._sum_stages(ii, jj)
                Gij = []
                Hij = []
                for kk in range(1, opts.n_s+1):
                    Gij.append(self.w.lambda_normal[ii,jj,kk])
                    Hij.append(sum_y_gap)
                    G_f, H_f = self._friction_pairs_stage_stage(
                        ii, jj, self._at_stage(ii,jj,kk), 1, sum_at)
                    Gij += G_f; Hij += H_f
                    if use_Y:
                        G_f, H_f = self._friction_pairs_stage_impulse(
                            ii, jj, self._at_stage(ii,jj,kk), 1)
                        Gij += G_f; Hij += H_f
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __fe_fe(self):
        opts = self.opts
        no_ii = opts.no_initial_impacts
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                use_Y = (jj != 1 or not no_ii) and not self._is_relaxed_oc()
                y_gap_lb = self.w.Y_gap[ii,jj] if use_Y else 0
                sum_y_gap = y_gap_lb + ca.sum2(self.w.y_gap[ii,jj,:].sym)
                sum_lambda = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)
                sum_at = self._sum_stages(ii, jj)
                Gij = [sum_lambda]
                Hij = [sum_y_gap]
                G_f, H_f = self._friction_pairs_stage_stage(ii, jj, sum_at, opts.n_s, sum_at)
                Gij += G_f; Hij += H_f
                if use_Y:
                    G_f, H_f = self._friction_pairs_stage_impulse(ii, jj, sum_at, opts.n_s)
                    Gij += G_f; Hij += H_f
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __standard(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                for kk in range(1, opts.n_s+1):
                    at = self._at_stage(ii, jj, kk)
                    Gij = [self.w.lambda_normal[ii,jj,kk].sym]
                    Hij = [self.w.y_gap[ii,jj,kk].sym]
                    G_f, H_f = self._friction_pairs_stage_stage(ii, jj, at, 1, at)
                    Gij += G_f; Hij += H_f
                    self.G.standard_comp[ii,jj,kk] = CConstraint(ca.vertcat(*Gij))
                    self.H.standard_comp[ii,jj,kk] = CConstraint(ca.vertcat(*Hij))


    @override
    def _generate_step_equilibration_constraints(self):
        opts = self.opts

        if not opts.use_fesd:
            return

        if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_MEAN:
            self._heuristic_mean()
        elif opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA:
            self._heuristic_diff()
        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED_SCALED:
            self._l2_relaxed_scaled()
        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
            self._l2_relaxed()
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT:
            self._direct()
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT_HOMOTOPY:
            raise NotImplementedError("Direct homotopy step-eq mode not currently implemented for CLS.")
        elif opts.step_equilibration == StepEquilibrationMode.LINEAR_COMPLEMENTARITY:
            raise NotImplementedError("MLCP formulation of step equilibration not yet supported for FESD-J.")

    @override
    def _get_eta(self, ii, jj):
        """
        Switch indicator eta_n built from the gap and contact force.
        Positive if no switch happens at the boundary between FE jj-1 and jj, zero otherwise.

        With friction the indicator additionally has to vanish at a stick/slip transition or a
        reversal of the sliding direction, otherwise FESD only adapts the step size at impacts and
        the friction switches are smeared over a finite element.
        """
        opts = self.opts
        rbp = self.rbp
        no_ii = opts.no_initial_impacts

        use_Y = (jj != 2 or not no_ii) and not self._is_relaxed_oc()
        sigma_c_B = self.w.Y_gap[ii,jj-1] if use_Y else 0
        sigma_c_B = sigma_c_B + ca.sum2(self.w.y_gap[ii,jj-1,:].sym)
        sigma_lambda_B = ca.sum2(self.w.lambda_normal[ii,jj-1,:].sym)

        Y_gap_F = self.w.Y_gap[ii,jj] if not self._is_relaxed_oc() else 0
        sigma_c_F = Y_gap_F + ca.sum2(self.w.y_gap[ii,jj,:].sym)
        sigma_lambda_F = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)

        nu = sigma_c_B*sigma_c_F + sigma_lambda_B*sigma_lambda_F
        if self.model.friction_exists:
            # A friction switch only matters while the contact is closed, so the friction indicator
            # is added to the gap sums rather than multiplied into them: out of contact the gap sums
            # are large and dominate, in contact they vanish and the friction terms decide.
            nu = nu*(sigma_c_B + sigma_c_F + self._friction_eta_term(ii, jj, use_Y))
        eta = 1
        for kk in range(nu.size()[0]):
            eta = eta*nu[kk]
        return eta

    def _friction_eta_term(self, ii, jj, use_impulse):
        """
        Per contact friction switch indicator, an `n_c` vector, zero exactly at a friction switch.

        Built from the same backward/forward sums as the normal part: `beta` (conic) or `beta_d`
        (polyhedral) detects the stick/slip transition, and the tangential velocity split (conic) or
        the per generator velocities (polyhedral) detect a reversal of the sliding direction. The
        tangential quantities have one entry per generator, so they are summed over each contact
        block to bring them back to `n_c`.
        """
        dims = self.dcs.dims

        def sigma(name, cap=None, back=False):
            idx = jj-1 if back else jj
            s = ca.sum2(getattr(self.w, name)[ii,idx,:].sym)
            include_cap = use_impulse if back else not self._is_relaxed_oc()
            if cap is not None and include_cap:
                s = s + getattr(self.w, cap)[ii,idx]
            return s

        def pi(name, cap=None):
            return sigma(name, cap, back=True)*sigma(name, cap, back=False)

        def per_contact(expr):
            """Collapse an n_tangents vector to n_c by summing each contact block."""
            n_t = self.variant.n_t
            return ca.vertcat(*[ca.sum1(expr[kk*n_t:(kk+1)*n_t]) for kk in range(dims.n_c)])

        if self._is_polyhedral():
            return pi("beta_d", "Beta_d") + per_contact(pi("delta_d", "Delta_d"))

        xi = pi("beta", "Beta")
        if self._switch_handling() != ConicModelSwitchHandling.PLAIN:
            xi = xi + per_contact(pi("p_vt", "P_vt")) + per_contact(pi("n_vt", "N_vt"))
        return xi


    @override
    def _warmstart_shift(self):
        """Warmstart the current problem by shifting one control interval"""
        raise NotImplementedError("Shift warmstarting not yet implemented for CLS")
