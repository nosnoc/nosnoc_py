from typing import Optional, List, override
from ..model import Cls as ClsModel, ClsDims
from ..dims import Dims
from .base import Base

import casadi as ca
import numpy as np


class ClsDcsDims(Dims):
    def __init__(self, parent: ClsDims):
        super().__init__(parent)
        self.n_lambda_normal = 0
        self.n_y_gap = 0


class Cls(Base):
    r"""
    FESD-J reformulation of a Complementarity Lagrangian System into a DCS.

    The contact forces are determined by the complementarity conditions

        0 <= lambda_normal  perp  y_gap >= 0,   y_gap = f_c(q),

    and, at the boundaries of the finite elements, the impulse equations determine either a state
    jump or the continuity of the velocities. `y_gap` and `Y_gap` are lifting variables for f_c(q),
    which keep the complementarity conditions linear in the variables.

    Note:
        Friction is not yet implemented, cf. `nosnoc.model.Cls`.
    """
    def __init__(self, model: ClsModel):
        self.dims = ClsDcsDims(model.dims)
        super().__init__(model)

    @override
    def _generate_variables(self):
        """Generate the required variables for the dcs"""
        dims = self.dims
        dims.n_lambda_normal = dims.n_c
        dims.n_y_gap = dims.n_c

        self.lambda_normal = ca.SX.sym("lambda_normal", dims.n_c)
        self.y_gap = ca.SX.sym("y_gap", dims.n_c)

        self.Lambda_normal = ca.SX.sym("Lambda_normal", dims.n_c)
        self.Y_gap = ca.SX.sym("Y_gap", dims.n_c)

        # Positive and negative parts of the restitution law residual. They lift the absolute value
        # in the aggregated impulse complementarity, cf. Eq. (A.2) of the FESD-J paper.
        self.P_vn = ca.SX.sym("P_vn", dims.n_c)
        self.N_vn = ca.SX.sym("N_vn", dims.n_c)

        self.z_alg = ca.vertcat(self.lambda_normal, self.y_gap)
        self.z_impulse = ca.vertcat(self.Lambda_normal, self.Y_gap, self.P_vn, self.N_vn)
        # Algebraics that appear in the right hand side of the CLS ODE.
        self.z_alg_f_x = self.lambda_normal

        self.z_all = ca.vertcat(self.z_alg, self.model.z)

    @override
    def _generate_expressions(self):
        """Generate the required equations and functions for the dcs"""
        model = self.model
        dims = self.dims
        J_n = model.J_normal

        self.f_x = ca.vertcat(model.v, model.inv_M@(model.f_v + J_n@self.lambda_normal))

        self.g_alg = self.y_gap - model.f_c

        v_post_impact = ca.SX.sym("v_post_impact", dims.n_q)
        v_pre_impact = ca.SX.sym("v_pre_impact", dims.n_q)

        g_impulse = [model.M@(v_post_impact - v_pre_impact) - J_n@self.Lambda_normal]
        g_impulse.append(self.Y_gap - model.f_c)
        
        for ii in range(dims.n_c):
            g_impulse.append(
                self.P_vn[ii] - self.N_vn[ii]
                - J_n[:,ii].T@(v_post_impact + model.e[ii]*v_pre_impact)
            )
        self.g_impulse = ca.vertcat(*g_impulse)

        
        self.f_x_fun = ca.Function('f_x', [model.x, model.z, self.z_alg_f_x, model.u, model.v_global, model.p], [self.f_x, model.f_q])
        self.f_q_fun = ca.Function('f_q', [model.x, model.z, model.u, model.v_global, model.p], [model.f_q])
        self.g_z_fun = ca.Function('g_z', [model.x, model.z, model.u, model.v_global, model.p], [model.g_z])
        self.g_alg_fun = ca.Function('g_alg', [model.x, model.z, self.z_alg, model.v_global, model.p], [self.g_alg])
        self.g_impulse_fun = ca.Function('g_impulse', [model.q, v_post_impact, v_pre_impact, self.z_impulse, model.v_global, model.p], [self.g_impulse])

        self.M_fun = ca.Function('M_fun', [model.x], [model.M])
        self.invM_fun = ca.Function('invM_fun', [model.x], [model.inv_M])
        self.f_c_fun = ca.Function('f_c_fun', [model.x], [model.f_c])
        self.J_normal_fun = ca.Function('J_normal_fun', [model.x], [J_n])

        self.g_path_fun = ca.Function('g_path', [model.x, model.z, model.u, model.v_global, model.p], [model.g_path])
        self.G_path_fun = ca.Function('G_path', [model.x, model.z, model.u, model.v_global, model.p], [model.G_path])
        self.H_path_fun = ca.Function('H_path', [model.x, model.z, model.u, model.v_global, model.p], [model.H_path])
        self.g_terminal_fun = ca.Function('g_terminal', [model.x, model.z, model.v_global, model.p_global], [model.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T', [model.x, model.z, model.v_global, model.p], [model.f_q_T])

        self.f_x_rk = ca.Function(
            'f_x_rk',
            [ca.vertcat(model.x, model.z, self.z_alg),
             ca.vertcat(model.u, model.v_global, model.p)],
            [self.f_x]
        )
        self.f_q_rk = ca.Function(
            'f_q_rk',
            [ca.vertcat(model.x, model.z, self.z_alg),
             ca.vertcat(model.u, model.v_global, model.p)],
            [model.f_q]
        )
        self.g_rk = ca.Function(
            'g_rk',
            [ca.vertcat(model.x, model.z, self.z_alg),
             ca.vertcat(model.u, model.v_global, model.p)],
            [ca.vertcat(model.g_z, self.g_alg)]
        )
