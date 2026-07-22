from typing import override

import casadi as ca
import numpy as np

from .base import Base
from vdx.vartypes import *

from ..nosnoc_types import RKRepresentation, CrossComplementarityMode, StepEquilibrationMode


class Cls(Base):
    r"""
    FESD-J discrete time problem (MPCC) for a Complementarity Lagrangian System.
    """

    def __init__(self, dcs, opts):
        super().__init__(dcs, opts)

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

            # Impulse variables at the finite element boundaries. They only exist where an impact is
            # allowed, i.e. from start_fe onwards.
            if opts.use_fesd:
                fe_range = range(start_fe, opts.N_finite_elements[ii-1]+1)
                self.w.Lambda_normal[ii,fe_range] = Primal("Lambda_normal", dims.n_c, lb=0.0, ub=opts.ub_Lambda_normal, init=opts.initial_Lambda_normal)
                self.w.P_vn[ii,fe_range] = Primal("P_vn", dims.n_c, lb=0.0, ub=opts.ub_P_vn, init=opts.initial_P_vn)
                self.w.N_vn[ii,fe_range] = Primal("N_vn", dims.n_c, lb=0.0, ub=opts.ub_N_vn, init=opts.initial_N_vn)
                self.w.Y_gap[ii,fe_range] = Primal("Y_gap", dims.n_c, lb=0.0, ub=opts.ub_Y_gap, init=opts.initial_Y_gap)

        self._handle_x_box_constraints()

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
            self.w.x[ii,range(1, opts.N_finite_elements[ii-1]+1),range(0,opts.n_s+rbp+1)] = Primal(f"x", dims.n_x,
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
        return ca.vertcat(
            self.w.Lambda_normal[ii,jj],
            self.w.Y_gap[ii,jj],
            self.w.P_vn[ii,jj],
            self.w.N_vn[ii,jj],
        )

    @override
    def _get_rk_stage_z(self, ii, jj, kk):
        # The stacked algebraic order (lambda_normal, y_gap) must match the dcs `z_alg` used to
        # build f_x_rk / f_q_rk / g_rk.
        if self.opts.rk_representation == RKRepresentation.INTEGRAL:
            return ca.vertcat(
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lambda_normal[ii,jj,kk],
                self.w.y_gap[ii,jj,kk],
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lambda_normal[ii,jj,kk],
                self.w.y_gap[ii,jj,kk],
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL_LIFT_X:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lambda_normal[ii,jj,kk],
                self.w.y_gap[ii,jj,kk],
            )

    # ------------------------------------------------------------------ transcription

    @override
    def _generate_direct_transcription_constraints(self):
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp

        x_0 = self.w.x[0,0,opts.n_s].sym
        z_0 = self.w.z[0,0,opts.n_s].sym
        self.g.z[0,0,opts.n_s] = Constraint(
            dcs.g_z_fun(x_0, z_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1)))

        x_prev = x_0  # right boundary of the previous FE, the pre impact state.
        for ii in range(1, opts.N_stages+1):
            s_sot = self._get_stage_sot(ii)
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                h = self._get_fe_h(ii, jj)
                u_i = self.w.u[ii]
                v_global = self.w.v_global[()]
                p = self._get_stage_parameters(ii)

                # ---- FESD-J boundary handling: continuity of q, impulse/jump for v ----
                
                q_prev = x_prev[0:dims.n_q]
                v_prev = x_prev[dims.n_q:]
                x_lbp = self.w.x[ii,jj,0].sym
                q_lbp = x_lbp[0:dims.n_q]
                v_lbp = x_lbp[dims.n_q:]
                self.g.q_continuity[ii,jj] = Constraint(q_prev - q_lbp)

                if opts.use_fesd and (jj != 1 or not opts.no_initial_impacts):
                    self.g.impulse[ii,jj] = Constraint(
                        dcs.g_impulse_fun(q_lbp, v_lbp, v_prev, self._build_z_impulse(ii,jj), v_global, p))
                    # Eq. (18): enforce the gap shortly after the impact to exclude spurious
                    # solutions with a zero impulse and a huge contact force.
                    if opts.eps_cls > 0:
                        step = opts.eps_cls if opts.fixed_eps_cls else h*opts.eps_cls
                        x_eps = ca.vertcat(q_lbp + step*v_lbp, v_lbp)
                        self.g.f_c_eps[ii,jj] = Constraint(dcs.f_c_fun(x_eps), lb=0.0, ub=np.inf)
                else:
                    self.g.v_continuity[ii,jj] = Constraint(v_lbp - v_prev)

                # ---- RK collocation, started from the (post impact) left boundary point ----
                z_ii_jj = self._build_z(ii, jj)
                prk_ii_jj = self._build_prk(ii, jj)
                x_end, q_end, dynamic, algebraic = self.rk.collocation_constraints(
                    x_lbp, z_ii_jj, prk_ii_jj, h, dcs.f_x_rk, dcs.f_q_rk, dcs.g_rk, sot=s_sot)
                for kk in range(1, opts.n_s+1):
                    self.g.dynamic[ii,jj,kk] = Constraint(dynamic[kk-1])
                    self.g.algebraic[ii,jj,kk] = Constraint(algebraic[kk-1])
                    self._rk_stage_path_constraints(ii, jj, kk)
                self.f += q_end

                x_ii_jj_end = self._get_x_end(ii, jj)
                if not self.rk.is_right_boundary_explicit():
                    self.g.dynamic[ii,jj,opts.n_s+1] = Constraint(x_end - x_ii_jj_end)
                    self.g.z[ii,jj,opts.n_s+1] = Constraint(
                        dcs.g_z_fun(x_ii_jj_end, self.w.z[ii,jj,opts.n_s+1].sym, u_i, v_global, p))
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

    # ------------------------------------------------------------------ complementarities

    @override
    def _generate_complementarity_constraints(self):
        opts = self.opts

        if opts.use_fesd:
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
                Gij = ca.vertcat(self.w.Lambda_normal[ii,jj], self.w.P_vn[ii,jj])
                Hij = ca.vertcat(
                    self.w.Y_gap[ii,jj] + self.w.P_vn[ii,jj] + self.w.N_vn[ii,jj],
                    self.w.N_vn[ii,jj],
                )
                self.G.impulse_comp[ii,jj] = CConstraint(Gij)
                self.H.impulse_comp[ii,jj] = CConstraint(Hij)

    def __stage_stage(self):
        opts = self.opts
        rbp = self.rbp
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                Gij = []
                Hij = []
                if jj != 1 or not opts.no_initial_impacts:
                    for kk in range(1, opts.n_s+1):
                        Gij.append(self.w.lambda_normal[ii,jj,kk])
                        Hij.append(self.w.Y_gap[ii,jj])
                for kk in range(1, opts.n_s+1):
                    for rr in range(1, opts.n_s+rbp+1):
                        Gij.append(self.w.lambda_normal[ii,jj,kk])
                        Hij.append(self.w.y_gap[ii,jj,rr])
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __fe_stage(self):
        opts = self.opts
        rbp = self.rbp
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lambda = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)
                Gij = []
                Hij = []
                if jj != 1 or not opts.no_initial_impacts:
                    Gij.append(sum_lambda)
                    Hij.append(self.w.Y_gap[ii,jj])
                for rr in range(1, opts.n_s+rbp+1):
                    Gij.append(sum_lambda)
                    Hij.append(self.w.y_gap[ii,jj,rr])
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __stage_fe(self):
        opts = self.opts
        
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                y_gap_lb = self.w.Y_gap[ii,jj] if (jj != 1 or not opts.no_initial_impacts) else 0
                sum_y_gap = y_gap_lb + ca.sum2(self.w.y_gap[ii,jj,:].sym)
                Gij = []
                Hij = []
                for kk in range(1, opts.n_s+1):
                    Gij.append(self.w.lambda_normal[ii,jj,kk])
                    Hij.append(sum_y_gap)
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))

    def __fe_fe(self):
        opts = self.opts
        no_ii = opts.no_initial_impacts
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                y_gap_lb = self.w.Y_gap[ii,jj] if (jj != 1 or not no_ii) else 0
                sum_y_gap = y_gap_lb + ca.sum2(self.w.y_gap[ii,jj,:].sym)
                sum_lambda = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)
                self.G.cross_comp[ii,jj] = CConstraint(sum_lambda)
                self.H.cross_comp[ii,jj] = CConstraint(sum_y_gap)

    def __standard(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                for kk in range(1, opts.n_s+1):
                    self.G.cross_comp[ii,jj,kk] = CConstraint(self.w.lambda_normal[ii,jj,kk].sym)
                    self.H.cross_comp[ii,jj,kk] = CConstraint(self.w.y_gap[ii,jj,kk].sym)

    # ------------------------------------------------------------------ step equilibration

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
            self.__l2_relaxed_scaled()
        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
            self.__l2_relaxed()
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT:
            self.__direct()
        elif opts.step_equilibration == StepEquilibrationMode.DIRECT_HOMOTOPY:
            raise NotImplementedError("Direct homotopy step-eq mode not currently implemented for CLS.")
        elif opts.step_equilibration == StepEquilibrationMode.LINEAR_COMPLEMENTARITY:
            raise NotImplementedError("MLCP formulation of step equilibration not yet supported for FESD-J.")

    def _get_eta(self, ii, jj):
        """
        Switch indicator eta_n built from the gap and contact force.
        Positive if no switch happens at the boundary between FE jj-1 and jj, zero otherwise.
        """
        opts = self.opts
        rbp = self.rbp
        no_ii = opts.no_initial_impacts

        sigma_c_B = self.w.Y_gap[ii,jj-1] if (jj != 2 or not no_ii) else 0
        sigma_c_B = sigma_c_B + ca.sum2(self.w.y_gap[ii,jj-1,:].sym)
        sigma_lambda_B = ca.sum2(self.w.lambda_normal[ii,jj-1,:].sym)

        sigma_c_F = self.w.Y_gap[ii,jj] + ca.sum2(self.w.y_gap[ii,jj,:].sym)
        sigma_lambda_F = ca.sum2(self.w.lambda_normal[ii,jj,:].sym)

        nu = sigma_c_B*sigma_c_F + sigma_lambda_B*sigma_lambda_F
        eta = 1
        for kk in range(nu.size()[0]):
            eta = eta*nu[kk]
        return eta

    def __l2_relaxed_scaled(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii, jj)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()]*ca.tanh(eta/opts.step_equilibration_sigma)*delta_h**2

    def __l2_relaxed(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii, jj)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()]*eta*delta_h**2

    def __direct(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii, jj)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.g.step_equilibration[ii,jj] = Constraint(eta*delta_h)
