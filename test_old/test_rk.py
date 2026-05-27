from nosnoc.rk import *
import casadi as ca
import numpy as np

# TODO(@anton) make this an actual test.
if __name__ == "__main__":
    #rk_representation = "integral"
    rk_representation = "differential_lift"
    if rk_representation == "integral":
        rk = IntegralRKRepresentation(4, RKScheme.RADAU_IIA)
        x = ca.SX.sym('x', 1)
        u = ca.SX.sym('u')
        f_x_sym = -x + u
        g_sym = ca.SX([])
        f_q_sym = x**2
        z = ca.vertcat(x)
        p = ca.vertcat(u)
        h = ca.SX.sym('h')
        x0 = ca.SX.sym('x0')
        f_x = ca.Function('f_x', [z, p], [f_x_sym])
        f_q = ca.Function('f_q', [z, p], [f_q_sym])
        g = ca.Function('g', [z, p], [g_sym])
        z = [ca.SX.sym(f"x_{ii}") for ii in range(1,5)]
        x_end, q_end, dynamic, algebraic = rk.collocation_constraints(x0, z, p, h, f_x, f_q, g)
    elif rk_representation == "differential":
        rk = DifferentialRKRepresentation(4, RKScheme.RADAU_IIA)
        x = ca.SX.sym('x', 1)
        u = ca.SX.sym('u')
        f_x_sym = -x + u
        g_sym = ca.SX([])
        f_q_sym = x**2
        z = ca.vertcat(x)
        p = ca.vertcat(u)
        h = ca.SX.sym('h')
        x0 = ca.SX.sym('x0')
        f_x = ca.Function('f_x', [ca.vertcat(x), p], [f_x_sym])
        f_q = ca.Function('f_q', [ca.vertcat(x), p], [f_q_sym])
        g = ca.Function('g', [ca.vertcat(x), p], [g_sym])
        z = [ca.SX.sym(f"v_{ii}") for ii in range(1,5)]
        x_end, q_end, dynamic, algebraic = rk.collocation_constraints(x0, z, p, h, f_x, f_q, g)
    elif rk_representation == "differential_lift":
        rk = LiftedDifferentialRKRepresentation(4, RKScheme.RADAU_IIA)
        x = ca.SX.sym('x', 1)
        u = ca.SX.sym('u')
        f_x_sym = -x + u
        g_sym = ca.SX([])
        f_q_sym = x**2
        z = ca.vertcat(x)
        p = ca.vertcat(u)
        h = ca.SX.sym('h')
        x0 = ca.SX.sym('x0')
        f_x = ca.Function('f_x', [ca.vertcat(x), p], [f_x_sym])
        f_q = ca.Function('f_q', [ca.vertcat(x), p], [f_q_sym])
        g = ca.Function('g', [ca.vertcat(x), p], [g_sym])
        z = [ca.vertcat(ca.SX.sym(f"v_{ii}"), ca.SX.sym(f"x_{ii}")) for ii in range(1,5)]
        x_end, q_end, dynamic, algebraic = rk.collocation_constraints(x0, z, p, h, f_x, f_q, g)
    import pdb; pdb.set_trace()
