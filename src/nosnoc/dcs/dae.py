from typing import Optional, List, override
from ..model import Dae as DaeModel, DaeDims
from ..dims import Dims
from .base import Base

import casadi as ca
import numpy as np


class Dae(Base):
    r"""
    Generating Function for the DAE.
    """
    def __init__(self, model:DaeModel):
        self.dims = model.dims
        super().__init__(model)

    @override
    def _generate_variables(self):
        """Generate the required variables for the dae. (none in this case)"""
        self.z_all = self.model.z

    @override
    def _generate_expressions(self):
        """Generate the required equations and functions for the dcs"""
        self.f_x = self.model.f_x

        self.g_alg = ca.vertcat([])

        self.f_x_fun = ca.Function('f_x', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.f_x, self.model.f_q])
        self.f_q_fun = ca.Function('f_q', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.f_q])
        self.g_z_fun = ca.Function('g_z', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_z])
        self.g_alg_fun = ca.Function('g_alg', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.g_alg])
        self.g_lp_stationarity_fun = ca.Function('g_lp_stationarity', [self.model.x, self.model.z, self.model.v_global, self.model.p], [self.g_alg])
        self.g_path_fun = ca.Function('g_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_path])
        self.g_terminal_fun  = ca.Function('g_terminal', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [self.model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [self.model.f_q_T])

        self.f_x_rk = ca.Function(
            'f_x_rk',
            [ca.vertcat(self.model.x, self.model.z),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.f_x]
        )
        self.f_q_rk = ca.Function(
            'f_q_rk',
            [ca.vertcat(self.model.x, self.model.z),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.model.f_q]
        )
        self.g_rk = ca.Function(
            'g_rk',
            [ca.vertcat(self.model.x, self.model.z),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z)]
        )
        self.g_rk_stationarity = ca.Function(
            'g_rk_stationarity',
            [ca.vertcat(self.model.x),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z)]
        )
        # TODO(@anton) implement
        # self.f_lsq_x_fun = ca.Function('f_lsq_x_fun',[self.model.x,self.model.x_ref,self.model.p],[self.model.f_lsq_x])
        # self.f_lsq_u_fun = ca.Function('f_lsq_u_fun',[self.model.u,self.model.u_ref,self.model.p],[self.model.f_lsq_u])
        # self.f_lsq_T_fun = ca.Function('f_lsq_T_fun',[self.model.x,self.model.x_ref_end,self.model.p_global],[self.model.f_lsq_T])

