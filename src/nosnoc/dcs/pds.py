from typing import Optional, List, override
from ..model import Pds as PdsModel, PdsDims
from ..dims import Dims
from .base import Base

import casadi as ca
import numpy as np


        
class Pds(Base):
    r"""
    Projected Dynamical System reformulation into a DCS
    """
    def __init__(self, model: PdsModel):
        self.dims= Dims(model.dims)
        super().__init__(model)

    @override
    def _generate_variables(self):
        
        dims = self.dims
        dims.n_lambda_normal = dims.n_c
        dims.n_y_gap = dims.n_c

        self.lambda_n = ca.SX.sym("lambda_normal", dims.n_c)
        self.y_gap = ca.SX.sym("y_gap", dims.n_c)

    @override
    def _generate_expressions(self):
        
        model = self.model
        dims = self.dims
        

        self.f_x = model.f + model.J_n @ self.lambda_n

        
        self.f_x_fun = ca.Function('f_x', [model.x, model.z, model.u, model.v_global, model.p], [self.f_x, model.f_q])
        self.f_q_fun = ca.Function('f_q', [model.x, model.z, model.u, model.v_global, model.p], [model.f_q])
        self.g_z_fun = ca.Function('g_z', [model.x, model.z, model.u, model.v_global, model.p], [model.g_z])
        self.g_alg_fun = ca.Function('g_alg', [model.x, model.z, self.z_alg, model.v_global, model.p], [self.g_alg])


        self.g_path_fun = ca.Function('g_path', [model.x, model.z, model.u, model.v_global, model.p], [model.g_path])
        self.G_path_fun = ca.Function('G_path', [model.x, model.z, model.u, model.v_global, model.p], [model.G_path])
        self.H_path_fun = ca.Function('H_path', [model.x, model.z, model.u, model.v_global, model.p], [model.H_path])
        self.g_terminal_fun = ca.Function('g_terminal', [model.x, model.z, model.v_global, model.p_global], [model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [model.x, model.z, model.v_global, model.p], [model.f_q_T])

       
        self.f_x_rk = ca.Function(
            'f_x_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lambda_n),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.f_x]
        )
        self.f_q_rk = ca.Function(
            'f_q_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lambda_n),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.model.f_q]
        )
        self.g_rk = ca.Function(
            'g_rk',
            [ca.vertcat(self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z, self.g_alg)]
        )
      