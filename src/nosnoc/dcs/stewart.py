from typing import Optional, List, override
from ..model import Pss as PssModel, PssDims
from ..dims import Dims
from .base import Base

import casadi as ca
import numpy as np

class StewartDims(Dims):
    def __init__(self, parent :PssDims):
        super().__init__(parent)
        self.n_theta = 0
        self.n_lambda = 0
        self.n_mu = 0


class Stewart(Base):
    r"""
    Stewart reformulation of a PSS to a DCS.
    """
    def __init__(self, model:PssModel):
        self.dims = StewartDims(model.dims)
        super().__init__(model)

    @override
    def _generate_variables(self):
        """Generate the required variables for the dcs"""
        self.dims.n_theta = sum(self.dims.n_f_sys)
        self.dims.n_lambda = self.dims.n_theta
        self.dims.n_mu = self.dims.n_sys

        self.theta_sys = list()
        self.lam_sys = list()
        self.mu_sys = list()
        for ii in range(self.dims.n_sys):
            self.theta_sys.append(ca.SX.sym(f"theta_{ii}", self.dims.n_f_sys[ii]))
            self.lam_sys.append(ca.SX.sym(f"lambda_{ii}", self.dims.n_f_sys[ii]))
            self.mu_sys.append(ca.SX.sym(f"mu_{ii}", 1))

        self.theta = ca.vertcat(*self.theta_sys)
        self.lam = ca.vertcat(*self.lam_sys)
        self.mu = ca.vertcat(*self.mu_sys)

        self.z_all = ca.vertcat(self.theta, self.lam, self.mu, self.model.z)

    @override
    def _generate_expressions(self):
        """Generate the required equations and functions for the dcs"""
        self.f_x = self.model.f_0

        for ii in range(self.dims.n_sys):
            self.f_x = self.f_x + self.model.F[ii]@self.theta_sys[ii]

        self.g_indicator = self.model.g_indicator

        g_lp_stationarity = []
        g_convex = []
        lam00_expr = []

        for ii in range(self.dims.n_sys):
            g_lp_stationarity.append(self.g_indicator[ii] - self.lam_sys[ii] + self.mu_sys[ii])
            g_convex.append(ca.sum(self.theta_sys[ii]) - 1)
            lam00_expr.append(self.g_indicator[ii] - ca.mmin(self.g_indicator[ii]))

        self.g_alg = ca.vertcat(*g_lp_stationarity, *g_convex)

        self.f_x_fun = ca.Function('f_x', [self.model.x, self.model.z, self.lam, self.theta, self.mu, self.model.u, self.model.v_global, self.model.p], [self.f_x, self.model.f_q])
        self.f_q_fun = ca.Function('f_q', [self.model.x, self.model.z, self.lam, self.theta, self.mu, self.model.u, self.model.v_global, self.model.p], [self.model.f_q])
        self.g_z_fun = ca.Function('g_z', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_z])
        self.g_alg_fun = ca.Function('g_alg', [self.model.x, self.model.z, self.lam, self.theta, self.mu, self.model.u, self.model.v_global, self.model.p], [self.g_alg])
        self.g_lp_stationarity_fun = ca.Function('g_lp_stationarity', [self.model.x, self.model.z, self.lam, self.mu, self.model.v_global, self.model.p], [*g_lp_stationarity])
        self.g_indicator_fun = ca.Function('g_indicator', [self.model.x, self.model.z, self.model.v_global, self.model.p], [*self.model.g_indicator])
        self.lam00_fun = ca.Function('lam00', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [*lam00_expr])
        self.g_path_fun = ca.Function('g_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_path])
        self.G_path_fun  = ca.Function('G_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.G_path])
        self.H_path_fun  = ca.Function('H_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.H_path])
        self.g_terminal_fun  = ca.Function('g_terminal', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [self.model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [self.model.f_q_T])

        self.f_x_rk = ca.Function(
            'f_x_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lam, self.theta, self.mu),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.f_x]
        )
        self.f_q_rk = ca.Function(
            'f_q_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lam, self.theta, self.mu),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.model.f_q]
        )
        self.g_rk = ca.Function(
            'g_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lam, self.theta, self.mu),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z, self.g_alg)]
        )
        self.g_rk_stationarity = ca.Function(
            'g_rk_stationarity',
            [ca.vertcat(self.model.x, self.model.z, self.lam, self.mu),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z, *g_lp_stationarity)]
        )
        # TODO(@anton) implement
        # self.f_lsq_x_fun = ca.Function('f_lsq_x_fun',[self.model.x,self.model.x_ref,self.model.p],[self.model.f_lsq_x])
        # self.f_lsq_u_fun = ca.Function('f_lsq_u_fun',[self.model.u,self.model.u_ref,self.model.p],[self.model.f_lsq_u])
        # self.f_lsq_T_fun = ca.Function('f_lsq_T_fun',[self.model.x,self.model.x_ref_end,self.model.p_global],[self.model.f_lsq_T])
