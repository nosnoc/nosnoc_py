from typing import override

import casadi as ca
import numpy as np

from .base import Base
from vdx.vartypes import *

from ..nosnoc_types import RKRepresentation, CrossComplementarityMode, StepEquilibrationMode

class Heaviside(Base):

    def __init__(self, dcs, opts):
        super().__init__(dcs, opts)

    @override
    def _create_variables(self):
        """Create Optimization Variables"""
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp


        # Use base class to create global variables
        self._create_global_variables()

        # Use base class to create initial x and z variables
        self._create_initial_variables()

        # Create initial Heaviside algebraics
        self.w.alpha[0,0,opts.n_s]      = Primal(f"alpha_0", dims.n_alpha,
                                                 lb=0.0,
                                                 ub=1.0,
                                                 init=0.5)
        self.w.lambda_n[0,0,opts.n_s]   = Primal(f"lambda_n_0", dims.n_lambda,
                                                 lb=0.0,
                                                 ub=np.inf,
                                                 init=0.5)
        self.w.lambda_p[0,0,opts.n_s]   = Primal(f"lambda_p_0", dims.n_lambda,
                                                 lb=0.0,
                                                 ub=np.inf,
                                                 init=0.5)
        # Create controls
        self._create_u()

        for ii in range(1,opts.N_stages+1):
            self._create_h(ii)  # Create timestep variables
            self._create_xvz(ii) # Create x and z variables

            # Create Heaviside algebraics
            self.w.alpha[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+1)]        = Primal(f"alpha_0", dims.n_alpha,
                                                                                                         lb=0.0,
                                                                                                         ub=1.0,
                                                                                                         init=opts.initial_alpha)
            self.w.lambda_n[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal(f"lambda_n_0", dims.n_lambda,
                                                                                                         lb=0.0,
                                                                                                         ub=np.inf,
                                                                                                         init=opts.initial_lambda_n)
            self.w.lambda_p[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal(f"lambda_p_0", dims.n_lambda,
                                                                                                         lb=0.0,
                                                                                                         ub=np.inf,
                                                                                                         init=opts.initial_lambda_p)
            if opts.step_equilibration == StepEquilibrationMode.LINEAR_COMPLEMENTARITY:
                self.w.B_max[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('B_max', dims.n_lambda,lb=-np.inf,ub=np.inf)
                self.w.pi_lambda_n[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('pi_lambda_n', dims.n_lambda,lb=-np.inf,ub=np.inf)
                self.w.pi_lambda_p[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('pi_lambda_p', dims.n_lambda,lb=-np.inf,ub=np.inf)
                self.w.lambda_lambda_n[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('lambda_lambda_n', dims.n_lambda,lb=0,ub=np.inf)
                self.w.lambda_lambda_p[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('lambda_lambda_p', dims.n_lambda,lb=0,ub=np.inf)
                self.w.eta[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('eta', dims.n_lambda,lb=0,ub=np.inf)
                self.w.nu[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('nu', 1,lb=0,ub=np.inf)

        # handle relaxing intermediate box constraints
        self._handle_x_box_constraints()

    @override
    def _generate_direct_transcription_constraints(self):
        """Create direct transcription constraints"""
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp

        z_rk_0 = self._get_initial_z()
        self.g.algebraic[0,0,opts.n_s] = Constraint(
            self.dcs.g_rk(z_rk_0,self._get_stage_parameters(1))
        )

        x_prev = self.w.x[0,0,opts.n_s].sym # last point of previous FE, needed for continuity conditions

        for ii in range(1, opts.N_stages+1):
            s_sot = self._get_stage_sot(ii)

            sum_h = 0
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                h = self._get_fe_h(ii,jj)
                z_ii_jj = self._build_z(ii,jj)
                prk_ii_jj = self._build_prk(ii,jj)
                x_end, q_end, dynamic, algebraic = self.rk.collocation_constraints(
                    x_prev,
                    z_ii_jj,
                    prk_ii_jj,
                    h,
                    dcs.f_x_rk,
                    dcs.f_q_rk,
                    dcs.g_rk,
                    sot=s_sot
                )
                for kk in range(1, opts.n_s+1):
                    self.g.dynamic[ii,jj,kk] = Constraint(dynamic[kk-1])
                    self.g.algebraic[ii,jj,kk] = Constraint(algebraic[kk-1])
                    self._rk_stage_path_constraints(ii,jj,kk)

                # TODO(@anton) implement euler cost integration
                self.f += q_end

                x_ii_jj_end = self._get_x_end(ii,jj)
                # Handle rbp
                if not self.rk.is_right_boundary_explicit():
                    self.g.dynamic[ii,jj,opts.n_s+1] = Constraint(x_end - x_ii_jj_end)
                    self.g.algebraic[ii,jj,opts.n_s+1] = Constraint(
                        dcs.g_rk_stationarity(
                            self._get_stage_end(ii,jj),
                            prk_ii_jj
                        )
                    )
                self._fe_path_constraints(ii,jj)
                x_prev = x_ii_jj_end
            self._numerical_time_constraints(ii)
            self._stage_path_constraints(ii)

        self._terminal_constraint()
        self._terminal_objective()

    def _get_initial_z(self):
        return ca.vertcat(
            self.w.x[0,0,self.opts.n_s],
            self.w.z[0,0,self.opts.n_s],
            self.w.alpha[0,0,self.opts.n_s],
            self.w.lambda_n[0,0,self.opts.n_s],
            self.w.lambda_p[0,0,self.opts.n_s],
        )

    def _get_stage_end(self,ii,jj):
        return ca.vertcat(
            self.w.x[ii,jj,self.opts.n_s+self.rbp],
            self.w.z[ii,jj,self.opts.n_s+self.rbp],
            self.w.alpha[ii,jj,self.opts.n_s+self.rbp],
            self.w.lambda_n[ii,jj,self.opts.n_s+self.rbp],
            self.w.lambda_p[ii,jj,self.opts.n_s+self.rbp],
        )


    @override
    def _get_rk_stage_z(self, ii, jj, kk):
        if self.opts.rk_representation == RKRepresentation.INTEGRAL:
            return ca.vertcat(
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.alpha[ii,jj,kk],
                self.w.lambda_n[ii,jj,kk],
                self.w.lambda_p[ii,jj,kk],
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.alpha[ii,jj,kk],
                self.w.lambda_n[ii,jj,kk],
                self.w.lambda_p[ii,jj,kk],
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL_LIFT_X:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.alpha[ii,jj,kk],
                self.w.lambda_n[ii,jj,kk],
                self.w.lambda_p[ii,jj,kk],
            )

    @override
    def _generate_complementarity_constraints(self):
        model = self.model
        opts = self.opts
        dims = self.dcs.dims
        alpha_0 = self.w.alpha[0,0,opts.n_s]
        lambda_n_0 = self.w.lambda_n[0,0,opts.n_s]
        lambda_p_0 = self.w.lambda_p[0,0,opts.n_s]

        # inital_comp
        self.G.initial_comp = CConstraint(ca.vertcat(lambda_n_0, lambda_p_0))
        self.H.initial_comp = CConstraint(ca.vertcat(alpha_0, 1-alpha_0))

        if opts.use_fesd:
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

    def __stage_stage(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lambda_n_prev = self.w.lambda_n[0,0,opts.n_s]
        lambda_p_prev = self.w.lambda_p[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                Gij = []
                Hij = []
                for rr in range(1, opts.n_s+1):
                    alpha_ijr = self.w.alpha[ii,jj,rr]
                    Gij.append(ca.vertcat(lambda_n_prev,lambda_p_prev))
                    Hij.append(ca.vertcat(alpha_ijr, 1-alpha_ijr))

                for kk in range(1,(opts.n_s + rbp)+1):
                    lambda_n_ijk = self.w.lambda_n[ii,jj,kk]
                    lambda_p_ijk = self.w.lambda_p[ii,jj,kk]
                    for rr in range(1, opts.n_s+1):
                        alpha_ijr = self.w.alpha[ii,jj,rr]
                        Gij.append(ca.vertcat(lambda_n_ijk, lambda_p_ijk))
                        Hij.append(ca.vertcat(alpha_ijr, 1-alpha_ijr))
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lambda_n_prev = self.w.lambda_n[ii,jj,opts.n_s+rbp]
                lambda_p_prev = self.w.lambda_p[ii,jj,opts.n_s+rbp]

    def __fe_stage(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lambda_n_prev = self.w.lambda_n[0,0,opts.n_s]
        lambda_p_prev = self.w.lambda_p[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_alpha = ca.sum2(self.w.alpha[ii,jj,:].sym)
                sum_not_alpha = ca.sum2(1 - self.w.alpha[ii,jj,:].sym)
                Gij = [lambda_n_prev, lambda_p_prev]
                Hij = [sum_alpha, sum_not_alpha]
                for kk in range(1,(opts.n_s + rbp)+1):
                    lambda_n_ijk = self.w.lambda_n[ii,jj,kk]
                    lambda_p_ijk = self.w.lambda_p[ii,jj,kk]
                    Gij.append(ca.vertcat(lambda_n_ijk, lambda_p_ijk))
                    Hij.append(ca.vertcat(sum_alpha, sum_not_alpha))

                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lambda_n_prev = self.w.lambda_n[ii,jj,opts.n_s+rbp]
                lambda_p_prev = self.w.lambda_p[ii,jj,opts.n_s+rbp]

    def __stage_fe(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lambda_n_prev = self.w.lambda_n[0,0,opts.n_s]
        lambda_p_prev = self.w.lambda_p[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lambda_n = lambda_n_prev + ca.sum2(self.w.lambda_n[ii,jj,:])
                sum_lambda_p = lambda_p_prev + ca.sum2(self.w.lambda_p[ii,jj,:])
                Gij = []
                Hij = []
                for kk in range(1, opts.n_s+1):
                    alpha_ijk = self.w.alpha[ii,jj,kk]
                    Gij.append(ca.vertcat(sum_lambda_n, sum_lambda_p))
                    Hij.append(ca.vertcat(alpha_ijk, 1-alpha_ijk))

                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lambda_n_prev = self.w.lambda_n[ii,jj,opts.n_s+rbp]
                lambda_p_prev = self.w.lambda_p[ii,jj,opts.n_s+rbp]

    def __fe_fe(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lambda_n_prev = self.w.lambda_n[0,0,opts.n_s]
        lambda_p_prev = self.w.lambda_p[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lambda_n = lambda_n_prev + ca.sum2(self.w.lambda_n[ii,jj,:])
                sum_lambda_p = lambda_p_prev + ca.sum2(self.w.lambda_p[ii,jj,:])
                sum_alpha = ca.sum2(self.w.alpha[ii,jj,:].sym)
                sum_not_alpha = ca.sum2(1 - self.w.alpha[ii,jj,:].sym)
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(sum_lambda_n, sum_lambda_p))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(sum_alpha, sum_not_alpha))
                lambda_n_prev = self.w.lambda_n[ii,jj,opts.n_s+rbp]
                lambda_p_prev = self.w.lambda_p[ii,jj,opts.n_s+rbp]

    def __standard(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                for kk in range(1, opts.n_s+1):
                    lambda_n_ijk = self.w.lambda_n[ii,jj,kk]
                    lambda_p_ijk = self.w.lambda_p[ii,jj,kk]
                    alpha_ijk = self.w.alpha[ii,jj,kk]
                    self.G.standard_comp[ii,jj,kk] = CConstraint(ca.vertcat(lambda_n_ijk, lambda_p_ijk))
                    self.H.standard_comp[ii,jj,kk] = CConstraint(ca.vertcat(alpha_ijk, 1-alpha_ijk))

    @override
    def _generate_step_equilibration_constraints(self):
        """Create step equilibration constraints"""
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp

        if not opts.use_fesd: # do nothing
            return

        if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_MEAN:
            self._heuristic_mean()

        if opts.step_equilibration == StepEquilibrationMode.HEURISTIC_DELTA:
            self._heuristic_diff()

        if opts.step_equilibration == StepEquilibrationMode.L2_RELAXED_SCALED:
            self.__l2_relaxed_scaled()

        elif opts.step_equilibration == StepEquilibrationMode.L2_RELAXED:
            self.__l2_relaxed()

        elif opts.step_equilibration == StepEquilibrationMode.DIRECT:
            self.__direct()

        elif opts.step_equilibration == StepEquilibrationMode.DIRECT_HOMOTOPY:
            raise NotImplementedError("Direct homotopy step-eq mode not currently implemented")

        if opts.step_equilibration == StepEquilibrationMode.LINEAR_COMPLEMENTARITY:
            self.__linear_complementarity()

    def _get_eta(self, ii, jj):
        assert ii >= 1, jj<=opts.N_stages
        assert jj >= 2, jj<=opts.N_finite_elements[ii-1]
        opts = self.opts
        rbp = self.rbp

        sigma_lambda_n_B = ca.sum2(self.w.lambda_n[ii,jj-1,:].sym)
        sigma_lambda_p_B = ca.sum2(self.w.lambda_p[ii,jj-1,:].sym)

        sigma_lambda_n_F = self.w.lambda_n[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lambda_n[ii,jj,:].sym)
        sigma_lambda_p_F = self.w.lambda_p[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lambda_p[ii,jj,:].sym)

        pi_lambda_n = sigma_lambda_n_B * sigma_lambda_n_F
        pi_lambda_p = sigma_lambda_p_B * sigma_lambda_p_F
        nu = pi_lambda_n + pi_lambda_p
        eta = 1
        for jjj in range(nu.size()[0]):
            eta = eta*nu[jjj]

        return eta

    def __l2_relaxed_scaled(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii,jj)
                eta_vec = ca.vertcat(eta_vec,eta)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()] * ca.tanh(eta/opts.step_equilibration_sigma) * delta_h**2

    def __l2_relaxed(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii,jj)
                eta_vec = ca.vertcat(eta_vec,eta)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()] * eta * delta_h**2

    def __direct(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                eta = self._get_eta(ii,jj)
                eta_vec = ca.vertcat(eta_vec,eta)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.g.step_equilibration[ii,jj] = Constraint(eta*delta_h)

    def __linear_complementarity(self):
        opts = self.opts
        rbp = self.rbp
        dims = self.dcs.dims
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                h0 = self.p.T[()]/(opts.N_stages*opts.N_finite_elements[ii-1])
                sigma_lambda_n_B = ca.sum2(self.w.lambda_n[ii,jj-1,:])
                sigma_lambda_p_B = ca.sum2(self.w.lambda_p[ii,jj-1,:])

                sigma_lambda_n_F = self.w.lambda_n[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lambda_n[ii,jj,:])
                sigma_lambda_p_F = self.w.lambda_p[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lambda_p[ii,jj,:])

                lambda_lambda_n = self.w.lambda_lambda_n[ii,jj]
                lambda_lambda_p = self.w.lambda_lambda_p[ii,jj]
                B_max = self.w.B_max[ii,jj]
                pi_lambda_n = self.w.pi_lambda_n[ii,jj]
                pi_lambda_p = self.w.pi_lambda_p[ii,jj]
                eta = self.w.eta[ii,jj]
                nu = self.w.nu[ii,jj]

                self.g.pi_lambda_n_or[ii,jj] = Constraint(
                    ca.vertcat(
                        pi_lambda_n-sigma_lambda_n_F,
                        pi_lambda_n-sigma_lambda_n_B,
                        sigma_lambda_n_F+sigma_lambda_n_B-pi_lambda_n
                    ),
                    lb=0,
                    ub=np.inf
                )
                self.g.pi_lambda_p_or[ii,jj] = Constraint(
                    ca.vertcat(
                        pi_lambda_p-sigma_lambda_p_F,
                        pi_lambda_p-sigma_lambda_p_B,
                        sigma_lambda_p_F+sigma_lambda_p_B-pi_lambda_p
                    ),
                    lb=0,
                    ub=np.inf
                )

                # kkt conditions for min B, B>=sigmaB, B>=sigmaF:
                kkt_max = ca.vertcat(
                    1-lambda_lambda_p-lambda_lambda_n,
                    B_max-pi_lambda_n,
                    B_max-pi_lambda_p,
                )
                self.g.kkt_max[ii,jj] = Constraint(
                    kkt_max,
                    lb=0.0,
                    ub=np.hstack([np.zeros(dims.n_lambda),np.inf*np.ones(dims.n_lambda),np.inf*np.ones(dims.n_lambda)])
                )

                self.G.step_eq_kkt_max[ii,jj] = CConstraint(ca.vertcat((B_max-pi_lambda_n),(B_max-pi_lambda_p)))
                self.H.step_eq_kkt_max[ii,jj] = CConstraint(ca.vertcat(lambda_lambda_n,lambda_lambda_p))

                # eta calculation
                eta_const = ca.vertcat(eta-pi_lambda_p,eta-pi_lambda_n,eta-pi_lambda_p-pi_lambda_n+B_max)
                self.g.eta_const[ii,jj] = Constraint(
                    eta_const,
                    lb=np.hstack([-np.inf*np.ones(dims.n_lambda),-np.inf*np.ones(dims.n_lambda),np.zeros(dims.n_lambda)]),
                    ub=np.hstack([np.zeros(dims.n_lambda),np.zeros(dims.n_lambda),np.inf*np.ones(dims.n_lambda)])
                    )

                self.g.nu_or[ii,jj] = Constraint(ca.vertcat(nu-eta,ca.sum(eta.sym)-nu),lb=0,ub=np.inf)

                # the actual step eq conditions
                M=self.p.T[()]/opts.N_stages
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                step_equilibration = ca.vertcat(
                    delta_h + (1/h0)*nu*M,
                    delta_h - (1/h0)*nu*M,
                )
                self.g.step_equilibration[ii,jj] = Constraint(step_equilibration,np.array([0,-np.inf]),np.array([np.inf,0]))
