from parameterized import parameterized
import unittest
import nosnoc as ns
import casadi as ca
import numpy as np

options = (
    (n_p_global, n_p_time_var)
    for n_p_global in range(10)
    for n_p_time_var in range(10)
)

X_REF = np.array([1, 0])
N_STAGES = 50
def get_options(**kwargs):
    default_opts = {
        "rk_scheme"         : ns.RKScheme.RADAU_IIA,
        "n_s"               : 1,
        "use_fesd"          : False,
        "N_stages"          : N_STAGES,
        "N_finite_elements" : 1,
        "T"                 : 1,
        "print_level"       : 0,
    }
    return ns.Options(**(default_opts | kwargs))

def get_model(n_p_global, n_p_time_var):
    q = ca.SX.sym('q', 1)
    v = ca.SX.sym('v', 1)
    x = ca.vertcat(q, v)
    x0 = np.array([0, 0])

    p_global = ca.SX.sym('p_global', n_p_global)
    p_time_var = ca.SX.sym('p_time_var', n_p_time_var)
    #end

    u = ca.SX.sym('u', 1)

    e = 0

    x_ref = X_REF
    Q = np.diag([200, 10])
    # Q = np.diag([0, 0])

    Q_T = np.diag([500, 300])
    R = np.diag([1])

    f_q = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u
    f_term = (x - x_ref).T @ Q_T @ (x - x_ref)

    # this one should be the same as f_term = 500*(q - 3)**2
    # f_term = 500*(q - bar)**2

    f_v = u
    M = np.eye(1)

    phi = q

    model =  ns.model.Cls(
        x = x,
        x0 = x0,
        u = u,
        M = M,
        f_v = f_v,
        e = e,
        f_q = f_q,
        f_q_T = f_term,
        f_c = phi,
        p_global = p_global,
        p_time_var =p_time_var,
    )
    return model

def solve_ocp(n_p_global, n_p_time_var):
    opts = get_options()
    model = get_model(n_p_global, n_p_time_var)
    solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    solver = ns.OcpSolver(model, opts, solver_opts)
    # buggy stuff
    solver.set_param("p_global", (), np.random.rand(n_p_global))
    solver.set_param("p_time_var", (range(1, N_STAGES + 1),), 3*np.ones((N_STAGES, n_p_time_var)))
    # end
    solver.solve()
    return solver

class ParameterTests(unittest.TestCase):

    @parameterized.expand(options)
    def test_parameters(self, n_p_global, n_p_time_var):
        solve_ocp(n_p_global, n_p_time_var)

if __name__ == "__main__":
    unittest.main()
