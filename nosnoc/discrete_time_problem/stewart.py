from typing import override

import casadi as ca
import numpy as np

from .base import Base
from vdx_py.vartypes import *

from ..nosnoc_types import RKRepresentation

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
            self._create_xz(ii) # Create x and z variables

            # Create Stewart algebraics
            self.w.lam[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)]   = Primal(f"lambda", dims.n_lambda,
                                                                                                  lb=0.0,
                                                                                                  ub=np.inf,
                                                                                                  init=1.0)
            self.w.theta[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal(f"theta", dims.n_theta,
                                                                                                  lb=0.0,
                                                                                                  ub=np.inf,
                                                                                                  init=1.0/dims.n_theta)
            self.w.mu[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)]    = Primal(f"mu", dims.n_mu,
                                                                                                  lb=-np.inf,
                                                                                                  ub=-np.inf)

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

        x_0 = self.w.x[0,0,opts.n_s]
        z_0 = self.w.z[0,0,opts.n_s]
        lam_0 = self.w.lam[0,0,opts.n_s]
        theta_0 = self.w.theta[0,0,opts.n_s]
        mu_0 = self.w.mu[0,0,opts.n_s]

        self.g.algebraic[0,0,opts.n_s] = Constraint(
            ca.vertcat(
                dcs.g_z_fun(x_0, z_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1)),
                dcs.g_alg_fun(x_0, z_0, lam_0, theta_0, mu_0, self.w.u[1], self.w.v_global[()], self._get_stage_parameters(1))
            ))

        x_prev = x_0 # last point of previous FE, needed for continuity conditions

        for ii in range(1, opts.N_stages+1):
            s_sot = self._get_stage_sot(ii)

            sum_h = 0
            for jj in range(1, opts.N_finite_elements[ii-1]):
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
                            self._get_rk_stage_z(self, ii, jj, opts.n_s+1),
                            prk_ii_jj
                        )
                    )
                self._numerical_time_constraints(ii,jj)
                self._fe_path_constraints(ii,jj)
                x_prev = x_ii_jj_end
            self._stage_path_constraints(ii)


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
        """Create complementarity constraints"""
        pass

    @override
    def _generate_step_equilibration_constraints(self):
        """Create step equilibration constraints"""
        pass
