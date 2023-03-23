from typing import Optional
from warnings import warn

import numpy as np
import casadi as ca
from nosnoc.utils import casadi_length
from nosnoc.model import NosnocModel
from nosnoc.dims import NosnocDims

class NosnocOcp:
    """
    allows to specify

    1) constraints of the form:
    lbu <= u <= ubu
    lbx <= x <= ubx
    lbv_global <= v_global <= ubv_global
    g_terminal(x_terminal) = 0
    g_path_comp(x)

    2) cost of the form:
    f_q(x, u)  -- integrated over the time horizon
    +
    f_terminal(x_terminal) -- evaluated at the end
    """

    def __init__(
            self,
            lbu: Optional[np.ndarray] = None,
            ubu: Optional[np.ndarray] = None,
            u_guess: Optional[np.ndarray] = None,
            lbx: Optional[np.ndarray] = None,
            ubx: Optional[np.ndarray] = None,
            f_q: ca.SX = ca.SX.zeros(1),
            g_path: ca.SX = ca.SX.zeros(0),
            lbg: Optional[np.ndarray] = None,
            ubg: Optional[np.ndarray] = None,
            g_path_comp: ca.SX = ca.SX.zeros(0, 2),
            f_terminal: ca.SX = ca.SX.zeros(1),
            g_terminal: ca.SX = ca.SX.zeros(0),
            lbv_global: Optional[np.ndarray] = None,
            ubv_global: Optional[np.ndarray] = None,
            v_global_guess: Optional[np.ndarray] = None,
    ):
        # TODO: Make everything more pythonic via optionals instead of empty arrays
        self.lbu: Optional[np.ndarray] = lbu
        self.ubu: Optional[np.ndarray] = ubu
        self.u_guess: Optional[np.ndarray] = u_guess
        self.lbx: np.ndarray = lbx
        self.ubx: np.ndarray = ubx
        self.f_q: ca.SX = f_q
        self.g_path: ca.SX = g_path
        self.lbg: np.ndarray = lbg
        self.ubg: np.ndarray = ubg
        self.g_path_comp: ca.SX = g_path_comp
        self.f_terminal: ca.SX = f_terminal
        self.g_terminal: ca.SX = g_terminal
        self.lbv_global: np.ndarray = lbv_global
        self.ubv_global: np.ndarray = ubv_global
        self.v_global_guess: np.ndarray = v_global_guess

    def preprocess_ocp(self, model: NosnocModel):
        dims: NosnocDims = model.dims
        self.g_terminal_fun = ca.Function('g_terminal_fun', [model.x, model.p, model.v_global],
                                          [self.g_terminal])
        self.f_q_T_fun = ca.Function('f_q_T_fun', [model.x, model.p, model.v_global],
                                     [self.f_terminal])
        self.f_q_fun = ca.Function('f_q_fun', [model.x, model.u, model.p, model.v_global],
                                   [self.f_q])
        self.g_path_fun = ca.Function('g_path_fun', [model.x, model.u, model.p, model.v_global], [self.g_path])

        # path complementarities
        if self.g_path_comp.shape[1] != 2:
            raise ValueError("path complementarities should be width 2")
        if self.g_path_comp.shape[0] != 0:
            warn("OCP: using path complementarities. Note that all expressions a, b need to be bound as 0 <= a, 0 <= b. This is not yet done automatically.")

        # Process complementarities into 3 categories:
        # rk_stage, ctrl_stage, global
        self.g_global_comp = ca.SX.zeros(0, 2)
        self.g_ctrl_comp = ca.SX.zeros(0, 2)
        self.g_stage_comp = ca.SX.zeros(0, 2)

        rk_stage_vars = ca.vertcat(model.x, model.z)
        control_stage_vars = ca.vertcat(model.u, model.p_time_var)

        for ii in range(self.g_path_comp.shape[0]):
            expr = self.g_path_comp[ii, :]
            if any(ca.which_depends(expr, rk_stage_vars)):
                self.g_stage_comp = ca.vertcat(self.g_stage_comp, expr)
            elif any(ca.which_depends(expr, control_stage_vars)):
                self.g_ctrl_comp = ca.vertcat(self.g_ctrl_comp, expr)
            else:
                self.g_global_comp = ca.vertcat(self.g_global_comp, expr)

        self.g_global_comp_fun = ca.Function('g_global_comp_fun', [model.p_global, model.v_global], [self.g_global_comp])
        self.g_ctrl_comp_fun = ca.Function('g_ctrl_comp_fun', [model.u, model.p, model.v_global], [self.g_ctrl_comp])
        self.g_rk_comp_fun = ca.Function('g_rk_comp_fun',
                                            [model.x, model.u, model.z_all, model.p, model.v_global],
                                            [self.g_stage_comp])

        # path constraints
        n_g_path = casadi_length(self.g_path)
        if self.lbg is None:
            self.lbg = np.zeros((n_g_path,))
        elif len(self.lbg) != n_g_path:
            raise ValueError("lbg and g_path have inconsistent shapes.")
        if self.ubg is None:
            self.ubg = np.zeros((n_g_path,))
        elif len(self.ubg) != n_g_path:
            raise ValueError("ubg and g_path have inconsistent shapes")

        # box constraints
        if self.lbx is None:
            self.lbx = -np.inf * np.ones((dims.n_x,))
        elif len(self.lbx) != dims.n_x:
            raise ValueError("lbx should be empty or of length n_x.")
        if self.ubx is None:
            self.ubx = np.inf * np.ones((dims.n_x,))
        elif len(self.ubx) != dims.n_x:
            raise ValueError("ubx should be empty or of length n_x.")

        # box constraints for u
        if self.lbu is None:
            self.lbu = -np.inf * np.ones((dims.n_u,))
        elif len(self.lbu) != dims.n_u:
            raise ValueError("lbu should be empty or of length n_u.")
        if self.ubu is None:
            self.ubu = np.inf * np.ones((dims.n_u,))
        elif len(self.ubu) != dims.n_u:
            raise ValueError("ubu should be empty or of length n_u.")
        if self.u_guess is None:
            self.u_guess = np.zeros((dims.n_u,))
        elif len(self.u_guess) != dims.n_u:
            raise ValueError("u_guess should be empty or of length n_u.")

        # global variables
        n_v_global = casadi_length(model.v_global)
        if self.lbv_global is None:
            self.lbv_global = -np.inf * np.ones((n_v_global,))
        if self.lbv_global.shape != (n_v_global,):
            raise Exception("lbv_global and v_global have inconsistent shapes.")

        if self.ubv_global is None:
            self.ubv_global = np.inf * np.ones((n_v_global,))
        if self.ubv_global.shape != (n_v_global,):
            raise Exception("ubv_global and v_global have inconsistent shapes.")

        if self.v_global_guess is None:
            self.v_global_guess = np.zeros((n_v_global,))
        if self.v_global_guess.shape != (n_v_global,):
            raise Exception("v_global_guess and v_global have inconsistent shapes.")
