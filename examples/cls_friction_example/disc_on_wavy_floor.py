import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc
A, K, R = 0.1, 2*np.pi/2.0, 0.2          # amplitude, wave number, radius
M_D, G  = 1.0, 9.81
J_D     = 0.5*M_D*R**2                    # moment of inertia of a disk

X0 = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

q = ca.SX.sym("q", 3)                     # x, y, theta
v = ca.SX.sym("v", 3)
x_sym = ca.vertcat(q, v)

s     = A*ca.sin(K*q[0])                  # floor
s_x   = ca.jacobian(s, q[0])              # slope for normalization of normal and tangential contact Jacobian, since ∇f_c = (−s', 1, 0)ᵀ and therefore ||∇f_c|| = √(1+s'²)
w     = 1/ca.sqrt(1 + s_x**2)             

f_c       = w*(q[1] - s) - R
J_normal  = ca.vertcat(-w*s_x, w,     0)
J_tangent = ca.vertcat( w,     w*s_x, R)


model = nosnoc.model.Cls(
    x=x_sym, x0=X0,
    M=np.diag([M_D, M_D, J_D]),
    f_v=ca.vertcat(0.0, -M_D*G, 0.0),
    f_c=f_c,
    J_normal=J_normal,
    J_tangent=J_tangent,
    e=0.0, mu=0.4,
    name="disc_on_wavy_floor"
)