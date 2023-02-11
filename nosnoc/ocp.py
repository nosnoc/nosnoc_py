import numpy as np
import casadi as ca
from nosnoc.utils import casadi_length, is_var_in_list
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
    g_path(x)

    2) cost of the form:
    f_q(x, u)  -- integrated over the time horizon
    +
    f_terminal(x_terminal) -- evaluated at the end
    """

    def __init__(
            self,
            lbu: np.ndarray = np.ones((0,)),
            ubu: np.ndarray = np.ones((0,)),
            lbx: np.ndarray = np.ones((0,)),
            ubx: np.ndarray = np.ones((0,)),
            f_q: ca.SX = ca.SX.zeros(1),
            g_path: ca.SX = ca.SX.zeros(0),
            lbg: np.ndarray = np.ones((0,)),
            ubg: np.ndarray = np.ones((0,)),
            g_path_comp: ca.SX = ca.SX.zeros(0, 2),
            f_terminal: ca.SX = ca.SX.zeros(1),
            g_terminal: ca.SX = ca.SX.zeros(0),
            lbv_global: np.ndarray = np.ones((0,)),
            ubv_global: np.ndarray = np.ones((0,)),
            v_global_guess: np.ndarray = np.ones((0,)),
    ):
        # TODO: not providing lbu, ubu should work as well!
        self.lbu: np.ndarray = lbu
        self.ubu: np.ndarray = ubu
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

        # Process complementarities into 3 categories
        self.g_global_comp = ca.SX.zeros(0, 2)
        self.g_ctrl_comp = ca.SX.zeros(0, 2)
        self.g_stage_comp = ca.SX.zeros(0, 2)
        for ii in range(self.g_path_comp.shape[0]):
            expr = self.g_path_comp[ii, :]
            expr_vars = ca.symvar(expr)
            rk_stage_vars = ca.symvar(model.x) + ca.symvar(model.z)
            control_stage_vars = ca.symvar(model.u) + ca.symvar(model.p_time_var)

            # TODO a better approach
            tightest = 2
            for var in expr_vars:
                if is_var_in_list(var, rk_stage_vars):
                    tightest = 0
                    break
                elif is_var_in_list(var, control_stage_vars):
                    tightest = 1

            if tightest == 2:
                self.g_global_comp = ca.vertcat(self.g_global_comp, expr)
            elif tightest == 1:
                self.g_ctrl_comp = ca.vertcat(self.g_ctrl_comp, expr)
            else:
                self.g_stage_comp = ca.vertcat(self.g_stage_comp, expr)

        self.g_global_comp_fun = ca.Function('g_global_comp_fun', [model.p_global, model.v_global], [self.g_global_comp])
        self.g_ctrl_comp_fun = ca.Function('g_ctrl_comp_fun', [model.u, model.p, model.v_global], [self.g_ctrl_comp])
        self.g_stage_comp_fun = ca.Function('g_stage_comp_fun',
                                            [model.x, model.u, model.z, model.p, model.v_global],
                                            [self.g_stage_comp])

        # path constraints
        n_g_path = casadi_length(self.g_path)
        if len(self.lbg) == 0:
            self.lbx = np.zeros((n_g_path,))
        elif len(self.lbg) != n_g_path:
            raise ValueError("lbg and g_path have inconsistent shapes.")
        if len(self.ubg) == 0:
            self.ubx = np.zeros((n_g_path,))
        elif len(self.ubg) != n_g_path:
            raise ValueError("ubg and g_path have inconsistent shapes")

        # box constraints
        if len(self.lbx) == 0:
            self.lbx = -np.inf * np.ones((dims.n_x,))
        elif len(self.lbx) != dims.n_x:
            raise ValueError("lbx should be empty or of length n_x.")
        if len(self.ubx) == 0:
            self.ubx = np.inf * np.ones((dims.n_x,))
        elif len(self.ubx) != dims.n_x:
            raise ValueError("ubx should be empty or of length n_x.")

        # global variables
        n_v_global = casadi_length(model.v_global)
        if len(self.lbv_global) == 0:
            self.lbv_global = -np.inf * np.ones((n_v_global,))
        if self.lbv_global.shape != (n_v_global,):
            raise Exception("lbv_global and v_global have inconsistent shapes.")

        if len(self.ubv_global) == 0:
            self.ubv_global = -np.inf * np.ones((n_v_global,))
        if self.ubv_global.shape != (n_v_global,):
            raise Exception("ubv_global and v_global have inconsistent shapes.")

        if len(self.v_global_guess) == 0:
            self.v_global_guess = -np.inf * np.ones((n_v_global,))
        if self.v_global_guess.shape != (n_v_global,):
            raise Exception("v_global_guess and v_global have inconsistent shapes.")
