"""
Minimal quasi-static OCP: ball 1 (velocity controlled) pushes ball 2 to a target position.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

R = 0.2                              # radius of both balls
X0 = np.array([-1.0, 0.0, 0.0, 0.0]) # [p1, p2]
P2_REF = np.array([1.0, 0.5])        # target of ball 2

p1 = ca.SX.sym("p1", 2)
p2 = ca.SX.sym("p2", 2)
u = ca.SX.sym("u", 2)
x = ca.vertcat(p1, p2)

model = nosnoc.model.Pds(
    x=x,
    u=u,
    x0=X0,
    f=ca.vertcat(u, 0, 0),                     # only ball 1 is actuated
    f_c=ca.sumsqr(p1 - p2) - (2*R)**2,         # balls may not overlap
    lbu=-np.ones(2), ubu=np.ones(2),
    f_q=ca.sumsqr(p2 - P2_REF) + 1e-2*ca.sumsqr(u),
    f_q_T=100*ca.sumsqr(p2 - P2_REF),
)

opts = nosnoc.Options(
    N_stages=30,
    N_finite_elements=1,
    n_s=1,
    rk_scheme=nosnoc.RKScheme.RADAU_IIA,
    use_fesd=False,
    T=4.0,
)
solver = nosnoc.OcpSolver(model, opts, nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions())
solver.solve()

x_res = solver.get("x")
plt.plot(x_res[:, 0], x_res[:, 1], "o-", label="ball 1")
plt.plot(x_res[:, 2], x_res[:, 3], "o-", label="ball 2")
plt.plot(*P2_REF, "kx", label="target")
plt.axis("equal")
plt.legend()
plt.show()
