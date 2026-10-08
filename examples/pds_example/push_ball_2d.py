
#Minimal Pds OCP: ball 1 (velocity controlled) pushes ball 2 to a target position.

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nosnoc

R = 0.2                              
X0 = np.array([-1.0, 0.0, 0.0, 0.0]) 
P2_REF = np.array([1.0, 0.5])        
T = 4.0
N_STAGES = 30

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
    N_stages=N_STAGES,
    N_finite_elements=1,
    n_s=1,
    rk_scheme=nosnoc.RKScheme.RADAU_IIA,
    use_fesd=False,
    T=T,
)
solver = nosnoc.OcpSolver(model, opts, nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions())
solver.solve()

x_res = solver.get("x")  


def circle(center, r=R):
    tt = np.linspace(0, 2*np.pi, 100)
    return center[0] + r*np.cos(tt), center[1] + r*np.sin(tt)


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid()
ax.plot(*circle(P2_REF), color="C3", alpha=0.4) 

(ball1,) = ax.plot([], [], "-", color="C0", lw=2, label="ball 1")
(ball2,) = ax.plot([], [], "-", color="C3", lw=2, label="ball 2")
(trail1,) = ax.plot([], [], "-", color="C0", alpha=0.3)
(trail2,) = ax.plot([], [], "-", color="C3", alpha=0.3)
ax.legend(loc="upper right")


def update(k):
    ball1.set_data(*circle(x_res[k, 0:2]))
    ball2.set_data(*circle(x_res[k, 2:4]))
    trail1.set_data(x_res[:k+1, 0], x_res[:k+1, 1])
    trail2.set_data(x_res[:k+1, 2], x_res[:k+1, 3])
    return ball1, ball2, trail1, trail2


anim = FuncAnimation(fig, update, frames=x_res.shape[0], interval=1000*T/N_STAGES, blit=True)
plt.show()
