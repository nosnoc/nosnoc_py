from typing import override

import casadi as ca
import numpy as np

from .base import Base
from vdx.vartypes import *

from ..nosnoc_types import RKRepresentation, CrossComplementarityMode, StepEquilibrationMode

class Stewart(Base):

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

        # Create initial Stewart algebraics
        self.w.lam[(0,0,opts.n_s)]   = Primal(f"lambda_0", dims.n_lambda,
                                              lb=0.0,
                                              ub=np.inf,
                                              init=1.0)
        self.w.theta[(0,0,opts.n_s)] = Primal(f"theta_0", dims.n_theta,
                                              lb=0,
                                              ub=np.inf,
                                              init=1.0/dims.n_theta)
        self.w.mu[(0,0,opts.n_s)]    = Primal(f"mu_0", dims.n_mu)

        # Create controls
        self._create_u()

        for ii in range(1,opts.N_stages+1):
            self._create_h(ii)  # Create timestep variables
            self._create_xvz(ii) # Create x and z variables

            # Create Stewart algebraics
            self.w.lam[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)]   = Primal(f"lambda", dims.n_lambda,
                                                                                                  lb=0.0,
                                                                                                  ub=np.inf,
                                                                                                  init=1.0)
            self.w.theta[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+1)] = Primal(f"theta", dims.n_theta,
                                                                                                  lb=0.0,
                                                                                                  ub=np.inf,
                                                                                                  init=1.0/dims.n_theta)
            self.w.mu[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)]    = Primal(f"mu", dims.n_mu,
                                                                                                  lb=-np.inf,
                                                                                                  ub=np.inf)
            if opts.step_equilibration == StepEquilibrationMode.LINEAR_COMPLEMENTARITY:
                self.w.B_max[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('B_max', dims.n_lambda ,lb=-np.inf,ub=np.inf)
                self.w.pi_theta[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('pi_theta', dims.n_theta ,lb=-np.inf,ub=np.inf)
                self.w.pi_lambda[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('pi_lambda', dims.n_lambda ,lb=-np.inf,ub=np.inf)
                self.w.theta_mult[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('theta_mult', dims.n_theta ,lb=0,ub=np.inf)
                self.w.lambda_mult[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('lambda_mult', dims.n_lambda ,lb=0,ub=np.inf)
                self.w.eta[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('eta', dims.n_lambda ,lb=0,ub=np.inf)
                self.w.nu[ii,range(2,opts.N_finite_elements[ii-1]+1)] = Primal('nu', 1, lb=0,ub=np.inf)

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

        x_0 = self.w.x[0,0,opts.n_s].sym
        z_0 = self.w.z[0,0,opts.n_s].sym
        lam_0 = self.w.lam[0,0,opts.n_s].sym
        theta_0 = self.w.theta[0,0,opts.n_s].sym
        mu_0 = self.w.mu[0,0,opts.n_s].sym

        self.g.algebraic[0,0,opts.n_s] = Constraint(
            ca.vertcat(
                dcs.g_z_fun(x_0, z_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1)),
                dcs.g_alg_fun(x_0, z_0, lam_0, theta_0, mu_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1))
            ))

        x_prev = x_0 # last point of previous FE, needed for continuity conditions

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
                    self.g.algebraic[ii,jj,opts.n_s+rbp] = Constraint(
                        dcs.g_rk_stationarity(
                            self._get_stage_end(ii, jj),
                            prk_ii_jj
                        )
                    )
                self._fe_path_constraints(ii,jj)
                x_prev = x_ii_jj_end
            self._numerical_time_constraints(ii)
            self._stage_path_constraints(ii)

        self._terminal_constraint()
        self._terminal_objective()

    def _get_stage_end(self,ii,jj):
        return ca.vertcat(
            self.w.x[ii,jj,self.opts.n_s+self.rbp],
            self.w.z[ii,jj,self.opts.n_s+self.rbp],
            self.w.lam[ii,jj,self.opts.n_s+self.rbp],
            self.w.theta[ii,jj,self.opts.n_s+self.rbp],
            self.w.mu[ii,jj,self.opts.n_s+self.rbp],
        )

    def _get_initial_z(self):
        return ca.vertcat(
            self.w.x[0,0,self.opts.n_s],
            self.w.z[0,0,self.opts.n_s],
            self.w.lam[0,0,self.opts.n_s],
            self.w.theta[0,0,self.opts.n_s],
            self.w.mu[0,0,self.opts.n_s],
        )

    def _get_rk_stage_z(self, ii, jj, kk):
        if self.opts.rk_representation == RKRepresentation.INTEGRAL:
            return ca.vertcat(
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lam[ii,jj,kk],
                self.w.theta[ii,jj,kk],
                self.w.mu[ii,jj,kk]
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lam[ii,jj,kk],
                self.w.theta[ii,jj,kk],
                self.w.mu[ii,jj,kk]
            )
        elif self.opts.rk_representation == RKRepresentation.DIFFERENTIAL_LIFT_X:
            return ca.vertcat(
                self.w.v[ii,jj,kk],
                self.w.x[ii,jj,kk],
                self.w.z[ii,jj,kk],
                self.w.lam[ii,jj,kk],
                self.w.theta[ii,jj,kk],
                self.w.mu[ii,jj,kk]
            )

    @override
    def _generate_complementarity_constraints(self):
        model = self.model
        opts = self.opts
        dims = self.dcs.dims
        lambda_0 = self.w.lam[0,0,opts.n_s]
        theta_0 = self.w.theta[0,0,opts.n_s]

        # inital_comp
        self.G.initial_comp = CConstraint(lambda_0)
        self.H.initial_comp = CConstraint(theta_0)

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
        lam_prev = self.w.lam[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                Gij = []
                Hij = []
                for rr in range(1, opts.n_s+1):
                    theta_ijr = self.w.theta[ii,jj,rr]
                    Gij.append(lam_prev)
                    Hij.append(theta_ijr)

                for kk in range(1,(opts.n_s + rbp)+1):
                    lam_ijk = self.w.lam[ii,jj,kk]
                    for rr in range(1, opts.n_s+1):
                        theta_ijr = self.w.theta[ii,jj,rr]
                        Gij.append(lam_ijk)
                        Hij.append(theta_ijr)
                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lam_prev = self.w.lam[ii,jj,opts.n_s + rbp]

    def __fe_stage(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lam_prev = self.w.lam[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_theta = ca.sum2(self.w.theta[ii,jj,:].sym)
                Gij = [lam_prev]
                Hij = [sum_theta]
                for kk in range(1,(opts.n_s + rbp)+1):
                    lam_ijk = self.w.lam[ii,jj,kk]
                    Gij.append(lam_ijk)
                    Hij.append(sum_theta)

                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lam_prev = self.w.lam[ii,jj,opts.n_s + rbp]

    def __stage_fe(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lam_prev = self.w.lam[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lam = lam_prev + ca.sum2(self.w.lam[ii,jj,:])
                Gij = []
                Hij = []
                for kk in range(1, opts.n_s+1):
                    theta_ijk = self.w.theta[ii,jj,kk]
                    Gij.append(sum_lam)
                    Hij.append(theta_ijk)

                self.G.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Gij))
                self.H.cross_comp[ii,jj] = CConstraint(ca.vertcat(*Hij))
                lam_prev = self.w.lam[ii,jj,opts.n_s + rbp]

    def __fe_fe(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        lam_prev = self.w.lam[0,0,opts.n_s]
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                sum_lam = lam_prev + ca.sum2(self.w.lam[ii,jj,:].sym)
                sum_theta = ca.sum2(self.w.theta[ii,jj,:].sym)
                self.G.cross_comp[ii,jj] = CConstraint(sum_lam)
                self.H.cross_comp[ii,jj] = CConstraint(sum_theta)
                lam_prev = self.w.lam[ii,jj,opts.n_s + rbp]

    def __standard(self):
        opts = self.opts
        dims = self.dcs.dims
        rbp = self.rbp
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                for kk in range(1, opts.n_s+1):
                    lam_ijk = self.w.lam[ii,jj,kk].sym
                    theta_ijk = self.w.theta[ii,jj,kk].sym
                    self.G.standard_comp[ii,jj,kk] = CConstraint(lam_ijk)
                    self.H.standard_comp[ii,jj,kk] = CConstraint(theta_ijk)

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

    def __l2_relaxed_scaled(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                sigma_lam_B = ca.sum2(self.w.lam[ii,jj-1,:])
                sigma_theta_B = ca.sum2(self.w.theta[ii,jj-1,:])

                sigma_lam_F = self.w.lam[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lam[ii,jj,:])
                sigma_theta_F = ca.sum2(self.w.theta[ii,jj,:])

                pi_lam = sigma_lam_B * sigma_lam_F
                pi_theta = sigma_theta_B * sigma_theta_F
                nu = pi_lam + pi_theta
                eta = 1
                for jjj in range(nu.size()[0]):
                    eta = eta*nu[jjj]

                eta_vec = ca.vertcat(eta_vec,eta)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()] * ca.tanh(eta/opts.step_equilibration_sigma) * delta_h**2

    def __l2_relaxed(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                sigma_lam_B = ca.sum2(self.w.lam[ii,jj-1,:])
                sigma_theta_B = ca.sum2(self.w.theta[ii,jj-1,:])

                sigma_lam_F = self.w.lam[ii,jj-1,opts.n_s + rbp] +ca.sum2(self.w.lam[ii,jj,:])
                sigma_theta_F = ca.sum2(self.w.theta[ii,jj,:])

                pi_lam = sigma_lam_B * sigma_lam_F
                pi_theta = sigma_theta_B * sigma_theta_F
                nu = pi_lam + pi_theta
                eta = 1
                for jjj in range(nu.size()[0]):
                    eta = eta*nu[jjj]

                eta_vec = ca.vertcat(eta_vec,eta)
                delta_h = self.w.h[ii,jj] - self.w.h[ii,jj-1]
                self.f += self.p.rho_h[()] * eta * delta_h**2

    def __direct(self):
        opts = self.opts
        rbp = self.rbp
        eta_vec = []
        for ii in range(1, opts.N_stages+1):
            for jj in range(2, opts.N_finite_elements[ii-1]+1):
                sigma_lam_B = ca.sum2(self.w.lam[ii,jj-1,:])
                sigma_theta_B = ca.sum2(self.w.theta[ii,jj-1,:])

                sigma_lam_F = self.w.lam[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lam[ii,jj,:])
                sigma_theta_F = ca.sum2(self.w.theta[ii,jj,:])

                pi_lam = sigma_lam_B * sigma_lam_F
                pi_theta = sigma_theta_B * sigma_theta_F
                nu = pi_lam + pi_theta
                eta = 1
                for jjj in range(nu.size()[0]):
                    eta = eta*nu[jjj]

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
                sigma_lambda_B = ca.sum2(self.w.lam[ii,jj-1,:])
                sigma_theta_B = ca.sum2(self.w.theta[ii,jj-1,:])

                sigma_lambda_F = self.w.lam[ii,jj-1,opts.n_s + rbp] + ca.sum2(self.w.lam[ii,jj,:])
                sigma_theta_F = ca.sum2(self.w.theta[ii,jj,:])

                lambda_mult = self.w.lambda_mult[ii,jj]
                theta_mult = self.w.theta_mult[ii,jj]
                B_max = self.w.B_max[ii,jj]
                pi_lambda = self.w.pi_lambda[ii,jj]
                pi_theta = self.w.pi_theta[ii,jj]
                eta = self.w.eta[ii,jj]
                nu = self.w.nu[ii,jj]

                self.g.pi_lambda_or[ii,jj] = Constraint(
                    ca.vertcat(
                        pi_lambda-sigma_lambda_F,
                        pi_lambda-sigma_lambda_B,
                        sigma_lambda_F+sigma_lambda_B-pi_lambda
                    ),
                    lb=0,
                    ub=np.inf
                )
                self.g.pi_theta_or[ii,jj] = Constraint(
                    ca.vertcat(
                        pi_theta-sigma_theta_F,
                        pi_theta-sigma_theta_B,
                        sigma_theta_F+sigma_theta_B-pi_theta
                    ),
                    lb=0,
                    ub=np.inf
                )

                # kkt conditions for min B, B>=sigmaB, B>=sigmaF:
                kkt_max = ca.vertcat(
                    1-theta_mult-lambda_mult,
                    B_max-pi_lambda,
                    B_max-pi_theta,
                )
                self.g.kkt_max[ii,jj] = Constraint(
                    kkt_max,
                    lb=0.0,
                    ub=np.hstack([np.zeros(dims.n_lambda),np.inf*np.ones(dims.n_lambda),np.inf*np.ones(dims.n_lambda)])
                )

                self.G.step_eq_kkt_max[ii,jj] = CConstraint(ca.vertcat((B_max-pi_lambda),(B_max-pi_theta)))
                self.H.step_eq_kkt_max[ii,jj] = CConstraint(ca.vertcat(lambda_mult,theta_mult))

                # eta calculation
                eta_const = ca.vertcat(eta-pi_theta,eta-pi_lambda,eta-pi_theta-pi_lambda+B_max)
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
