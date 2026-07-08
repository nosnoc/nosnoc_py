from typing import Optional, List, override
from ..model import Pss as PssModel, PssDims
from ..dims import Dims
from .base import Base

import casadi as ca
import numpy as np

class HeavisideDims(Dims):
    def __init__(self, parent :PssDims):
        super().__init__(parent)
        self.n_alpha = 0
        self.n_lambda = 0

class Heaviside(Base):
    r"""
    Heaviside step reformulation of a PSS or Heaviside step model to a DCS
    """
    def __init__(self, model:PssModel):
        self.dims = HeavisideDims(model.dims)
        super().__init__(model)

    @override
    def _generate_variables(self):
        self.dims.n_alpha = sum(self.dims.n_c_sys)
        self.dims.n_lambda = self.dims.n_alpha

        # TODO(@anton) implement automatic lifting, that is lifting $\theta(\alpha)$

        self.alpha_sys = list()
        self.lambda_n_sys = list()
        self.lambda_p_sys = list()
        for ii in range(self.dims.n_sys):
            self.alpha_sys.append(ca.SX.sym(f"alpha_{ii}", self.dims.n_c_sys[ii]))
            self.lambda_n_sys.append(ca.SX.sym(f"lambda_n_{ii}", self.dims.n_c_sys[ii]))
            self.lambda_p_sys.append(ca.SX.sym(f"lambda_p_{ii}", self.dims.n_c_sys[ii]))

        self.alpha = ca.vertcat(*self.alpha_sys)
        self.lambda_n = ca.vertcat(*self.lambda_n_sys)
        self.lambda_p = ca.vertcat(*self.lambda_p_sys)

        self.z_all = ca.vertcat(self.alpha, self.lambda_n, self.lambda_p, self.model.z)

    @override
    def _generate_expresions(self):
        """Generate the required equations and functions for the dcs"""
        self.f_x = self.model.f_0


        # TODO(@anton) there has to be a better way to do this!
        for ii in range(self.dims.n_sys):
            theta_ii = ca.SX.ones()
            for jj in range(self.model.S.shape[0]):
                for kk in range(self.model.S.shape[1]):
                    if self.model.S[jj,kk] != 0:
                        theta_ii[jj] *= (0.5*(1-self.model.S[jj,kk])+self.model.S[jj,kk]*self.alpha_sys[ii][kk])

            self.f_x += self.model.F[ii]@theta_ii

        self.g_indicator = self.model.g_indicator

        g_lp_stationarity = []
        lam00_n_expr = []
        lam00_p_expr = []
        for ii in range(self.dims.n_sys):
            g_lp_stationarity.append(self.model.c[ii] - self.lambda_p_sys[ii] + self.lambda_n_sys[ii])
            lam00_n_expr.append(-ca.min(self.model.c[ii], 0))
            lam00_p_expr.append(ca.max(self.model.c[ii],0))

        self.g_alg = ca.vertcat(*g_lp_stationarity)

        self.f_x_fun = ca.Function('f_x', [self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p, self.model.u, self.model.v_global, self.model.p], [self.f_x, self.model.f_q])
        self.f_q_fun = ca.Function('f_q', [self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p, self.model.u, self.model.v_global, self.model.p], [self.model.f_q])
        self.g_z_fun = ca.Function('g_z', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_z])
        self.g_alg_fun = ca.Function('g_alg', [self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p, self.model.u, self.model.v_global, self.model.p], [self.g_alg])
        self.g_lp_stationarity_fun = ca.Function('g_lp_stationarity', [self.model.x, self.model.z, self.lambda_n, self.lambda_p, self.model.v_global, self.model.p], [*g_lp_stationarity])
        self.g_indicator_fun = ca.Function('g_indicator', [self.model.x, self.model.z, self.model.v_global, self.model.p], [*self.model.g_indicator])
        self.lam00_fun = ca.Function('lam00', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [*lam00_expr])
        self.g_path_fun = ca.Function('g_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.g_path])
        self.G_path_fun  = ca.Function('G_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.G_path])
        self.H_path_fun  = ca.Function('H_path', [self.model.x, self.model.z, self.model.u, self.model.v_global, self.model.p], [self.model.H_path])
        self.g_terminal_fun  = ca.Function('g_terminal', [self.model.x, self.model.z, self.model.v_global, self.model.p_global], [self.model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [self.model.x, self.model.z, self.model.v_global, self.model.p], [self.model.f_q_T])

        self.f_x_rk = ca.Function(
            'f_x_rk',
            [ca.vertcat(self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.f_x]
        )
        self.f_q_rk = ca.Function(
            'f_q_rk',
            [ca.vertcat(self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [self.model.f_q]
        )
        self.g_rk = ca.Function(
            'g_rk',
            [ca.vertcat(self.model.x, self.model.z, self.alpha, self.lambda_n, self.lambda_p),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z, self.g_alg)]
        )
        self.g_rk_stationarity = ca.Function(
            'g_rk',
            [ca.vertcat(self.model.x, self.model.z, self.lambda_n, self.lambda_p),
             ca.vertcat(self.model.u, self.model.v_global, self.model.p)],
            [ca.vertcat(self.model.g_z, *g_lp_stationarity)]
        )
