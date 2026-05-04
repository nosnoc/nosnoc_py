from typing import override

from vdx_py.vector import *
import casadi as ca
import numpy as np

from .base import Base


class Stewart(Base):

    def __init__(self, dcs, opts):
        super().__init__(dcs, opts)

    def _create_parameters(self):
        self.p.rho_h[()] = Parameter("rho_h", 1, val=self.opts.rho_h)
        self.p.rho_terminal[()] = Parameter("rho_terminal", 1, val=self.opts.rho_terminal)
        self.p.T[()] = Parameter("T", 1, val=self.opts.T)
        self.p.p_global[()] = Parameter("p_global", self.dcs.dims.n_p_global, val=self.model.p_global_val)
        for ii in range(self.opts.N_stages):
            self.p.p_time_var[ii] = Parameter(f"p_time_var_{ii}", self.dcs.dims.n_p_time_var, val=self.model.p_time_var_val)

    @override
    def _create_variables(self):
        """Create Optimization Variables"""
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims

        self.w.v_global[()] = Primal("v_global",
                                     dims.n_v_global,
                                     init=model.v0_global,
                                     lb=model.lbv_global,
                                     ub=model.ubv_global)
        if opts.use_speed_of_time_variables and not opts.local_speed_of_time_variable:
            self.w.sot[()] = Primal("sot",
                                     1,
                                     init=opts.s_sot0,
                                     lb=model.s_sot_min,
                                     ub=model.s_sot_max)
        if opts.time_optimal_problem:
            self.w.T_final[()] = Primal("T_final", 1, lb=opts.T_final_min, ub=opts.T_final_max, init=opts.T)
            self.f += self.w.T_final[()]

        self.w.x[(0,0,opts.n_s)]     = Primal(f"x_0", dims.n_x,
                                              lb=model.x0,
                                              ub=model.x0,
                                              init=model.x0)
        self.w.z[(0,0,opts.n_s)]     = Primal(f"z_0", dims.n_z,
                                              lb=model.z0,
                                              ub=model.z0,
                                              init=model.z0)
        self.w.lam[(0,0,opts.n_s)]   = Primal(f"lam_0", dims.n_lambda,
                                              lb=0.0,
                                              ub=np.inf,
                                              init=1.0)
        self.w.theta[(0,0,opts.n_s)] = Primal(f"theta_0", dims.n_theta,
                                              lb=0,
                                              ub=np.inf,
                                              init=1.0/dims.n_theta)
        self.w.mu[(0,0,opts.n_s)]    = Primal(f"mu_0", dims.n_mu)

        for ii in range(1,opts.N_stages+1):
            self.w.u[ii] = Primal(f"u_{ii}", dims.n_u,
                                    lb=model.lbu,
                                    ub=model.ubu,
                                    init=model.u0)
            h0 = opts.h_k[ii-1]
            if opts.use_fesd:
                ubh = (1 + opts.gamma_h) * h0 # upper bound for FE length
                lbh = (1 - opts.gamma_h) * h0 # lower bound for FE length
                if opts.time_rescaling() and not opts.use_speed_of_time_variables:
                    # if only time_rescaling is true, speed of time and step size all lumped together, e.g., \hat{h}_{k,i} = s_n * h_{k,i}, hence the bounds need to be extended.
                    ubh = ubh*opts.s_sot_max;
                    lbh = lbh/opts.s_sot_min;
                elif opts.time_optimal_problem:
                    ubh = ubh*(opts.T_final_max/opts.T);
                    lbh = lbh/((opts.T_final_min+eps)/opts.T);end
                    
            for jj in range(1,opts.N_finite_elements[ii-1]+1):
                if opts.use_fesd:
                    self.w.h[ii,jj] = Primal(f"h_{ii}_{jj}", 1,
                                             lb=lbh,
                                             ub=ubh,
                                             init=h0)
                for kk in range(1,opts.n_s+1):
                    self.w.x[ii,jj,kk] = Primal(f"x_{ii}_{jj}_{kk}", dims.n_x,
                                                  lb=model.lbx,
                                                  ub=model.ubx,
                                                  init=model.x0)
                    self.w.z[ii,jj,kk] = Primal(f"z_{ii}_{jj}_{kk}", dims.n_z,
                                                  lb=model.lbz,
                                                  ub=model.ubz,
                                                  init=model.z0)
                    self.w.lam[ii,jj,kk] = Primal(f"lambda_{ii}_{jj}_{kk}", dims.n_lambda,
                                                    lb=0.0,
                                                    ub=np.inf,
                                                    init=1.0)
                    self.w.theta[ii,jj,kk] = Primal(f"theta_{ii}_{jj}_{kk}", dims.n_theta,
                                                      lb=0.0,
                                                      ub=np.inf,
                                                      init=1.0/dims.n_theta)
                    self.w.mu[ii,jj,kk] = Primal(f"mu_{ii}_{jj}_{kk}", dims.n_mu,
                                                   lb=-np.inf,
                                                   ub=-np.inf)
                if not self.rk.is_right_boundary_explicit():
                    self.w.x[(ii,jj)] =  Primal(f"x_{ii}_{jj}", dims.n_x,
                                                  lb=model.lbx,
                                                  ub=model.ubx,
                                                  init=model.x0)



    @override
    def _generate_direct_transcription_constraints(self):
        """Create direct transcription constraints"""
        pass

    @override
    def _generate_complementarity_constraints(self):
        """Create complementarity constraints"""
        pass

    @override
    def _generate_step_equilibration_constraints(self):
        """Create step equilibration constraints"""
        pass
