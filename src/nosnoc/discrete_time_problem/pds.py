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
        self.__apply_time_stepping_defaults(opts)
        super().__init__(dcs, opts)



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
        opts = self.opts
        dcs = self.dcs
        model = self.model
        dims = self.dcs.dims
        rbp = self.rbp

        x_0 = self.w.x[0,0,opts.n_s].sym
        z_0 = self.w.z[0,0,opts.n_s].sym
        
        

        for ii in range(1, opts.N_stages+1):
            for jj in range(1, opts.N_fe+1):
                for kk in range(1, opts.n_s+1):
                      self.g.path[ii,jj,kk] = Constraint(
                                self.dcs.g_alg(self.x, self.z, self.lambda_n, self.v_global, self.p),
                                lb=self.model.lbg_path,
                                ub=self.model.ubg_path, #
                            )


        self._terminal_constraint()
        self._terminal_objective()
        self._terminal_numerical_time_constraints()


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
            for jj in range(1, opts.N_finite_elements + 1):
                for kk in range(1, opts.n_s+1):
                    self.G.standard_comp[ii,jj,kk] = CConstraint(self.w.lambda_normal[ii,jj,kk].sym)
                    self.H.standard_comp[ii,jj,kk] = CConstraint(self.w.y_gap[ii,jj,kk].sym)


