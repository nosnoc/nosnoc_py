from typing import override
from warnings import warn

import casadi as ca
import numpy as np

from .base import Base
from vdx.vartypes import *

from ..nosnoc_types import RKRepresentation, CrossComplementarityMode, StepEquilibrationMode, ClsDiscretization, RKScheme


class Pds(Base):
    r"""
    Discrete time problem (MPCC) for a Projected Dynamical System.
    """

    def __init__(self, dcs, opts):
        super().__init__(dcs, opts)



    @override
    def _create_variables(self):
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp
        

        self._create_global_variables()
        self._create_initial_variables()
        self._create_speed_of_time_variables()
        self._create_u()

        for ii in range(1, opts.N_stages+1):
            self._create_h(ii)
            self._create_xvz(ii)

            
            self.w.lambda_normal[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+1)] = Primal(
                "lambda_normal", dims.n_c, lb=0.0, ub=opts.ub_lambda_normal, init=opts.initial_lambda_normal)
            self.w.y_gap[ii,range(1,opts.N_finite_elements[ii-1]+1),range(1,opts.n_s+rbp+1)] = Primal(
                "y_gap", dims.n_c, lb=0.0, ub=opts.ub_y_gap, init=opts.initial_y_gap)

        
            
        self._handle_x_box_constraints()

   
    

    @override
    def _get_rk_stage_z(self, ii, jj, kk):
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

    @override
    def _generate_direct_transcription_constraints(self):
        opts, dcs = self.opts, self.dcs
        x_prev = self.w.x[0,0,opts.n_s].sym
        
        for ii in range(1, opts.N_stages+1):
            s_sot = self._get_stage_sot(ii)
            for jj in range(1, opts.N_finite_elements[ii-1]+1):
                h = self._get_fe_h(ii, jj)
                x_end, q_end, dynamic, algebraic = self.rk.collocation_constraints(
                    x_prev, self._build_z(ii, jj), self._build_prk(ii, jj), h,
                    dcs.f_x_rk, dcs.f_q_rk, dcs.g_rk, sot=s_sot)
                
                for kk in range(1, opts.n_s+1):
                    self.g.dynamic[ii,jj,kk]  = Constraint(dynamic[kk-1])     
                    self.g.algebraic[ii,jj,kk] = Constraint(algebraic[kk-1])   

                    self._rk_stage_path_constraints(ii, jj, kk)

                self.f += q_end                                                
                x_ii_jj_end = self._get_x_end(ii, jj)

                if not self.rk.is_right_boundary_explicit():
                    self.g.dynamic[ii,jj,opts.n_s+1] = Constraint(x_end - x_ii_jj_end)
                    
                self._fe_path_constraints(ii, jj)
                x_prev = x_ii_jj_end                                         
            self._numerical_time_constraints(ii)
            self._stage_path_constraints(ii)
        self._terminal_constraint()


    @override
    def _generate_complementarity_constraints(self):
        opts = self.opts

        if opts.use_fesd:
            
           
            if opts.cross_comp_mode == CrossComplementarityMode.STAGE_STAGE:
                raise NotImplementedError("Only implicit Euler for now")
            elif opts.cross_comp_mode == CrossComplementarityMode.FE_STAGE:
                raise NotImplementedError("Only implicit Euler for now")
            elif opts.cross_comp_mode == CrossComplementarityMode.STAGE_FE:
                raise NotImplementedError("Only implicit Euler for now")
            elif opts.cross_comp_mode == CrossComplementarityMode.FE_FE:
                raise NotImplementedError("Only implicit Euler for now")
        else:
            self.__standard()

    def __standard(self):
        opts = self.opts
        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_finite_elements[ii-1] + 1):
                for kk in range(1, opts.n_s+1):
                    self.G.standard_comp[ii,jj,kk] = CConstraint(self.w.lambda_normal[ii,jj,kk].sym)
                    self.H.standard_comp[ii,jj,kk] = CConstraint(self.w.y_gap[ii,jj,kk].sym)


    @override
    def _generate_step_equilibration_constraints(self): #TODO @Stefan: implement FESD
        return

    @override
    def _get_eta(self, ii, jj):
        return

    @override
    def _warmstart_shift(self):
        """Warmstart the current problem by shifting one control interval"""
        raise NotImplementedError("Shift warmstarting not yet implemented for CLS")



