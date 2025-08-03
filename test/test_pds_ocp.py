import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from nosnoc.nosnoc_types import DcsMode, CrossComplementarityMode
from nosnoc.model import NosnocModel
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.solver import NosnocSolver
from nosnoc.ocp import NosnocOcp

# 1. Define state and control
n_x = 2
n_u = 2
x = ca.SX.sym('x', n_x)
u = ca.SX.sym('u', n_u)
fontsize = 18

# 2. Bounds and initial values
lbx = np.array([-np.inf, -np.inf])
ubx = np.array([np.inf, np.inf])
x0 = np.array([1.0, 5.0])
x_target = np.array([0.0, -4.0])

lbu = np.array([-1.0, -1.0])
ubu = np.array([1.0, 1.0])
u_guess = np.array([0.0, 0.0])

# 3. Quadratic constraint
P = np.array([[1/4, 0], [0, 1/16]])
c_pds = ca.mtimes([x.T, P, x]) - 1  # x'*P*x - 1

# 4. Nonlinear dynamics
f_unconstrained = [ca.vertcat(
    -0.2 * (x[0] + 1)**2,
    -0.4 * (x[1] + 3)
) + u]

# 5. Cost function (example: quadratic control cost)
R = ca.diagcat(1.0, 1.0)  # Control cost matrix
f_q = ca.mtimes([u.T, R, u])  # Quadratic cost on control input
Q_T = ca.diagcat(0.0, 0.0)  # Terminal cost matrix
f_q_T = 0.5*ca.mtimes([(x-x_target).T, Q_T, (x-x_target)])  # Terminal cost on state
g_terminal = x - x_target  # Terminal constraint

# 6. Build the OCP object
ocp = NosnocOcp(
    lbu=lbu,
    ubu=ubu,
    u_guess=u_guess,
    lbx=lbx,
    ubx=ubx,
    f_q=f_q,
    f_terminal=f_q_T,
    g_terminal=g_terminal,
)

# 7. Build the model
model = NosnocModel(
    x=x,
    u=u,
    x0=x0,
    f_unconstrained=f_unconstrained,
    c_pds=c_pds,
)


# 8. Set up options
T = 10.0
opts = NosnocOpts(
    dcs_mode=DcsMode.PDS,
    n_s=3,
    N_stages=10,
    terminal_time=T,
    use_fesd=True,
    print_level=2,
    cross_comp_mode= CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER_PDS,
)


# 10. Create and solve the OCP
solver = NosnocSolver(opts, model,ocp)
results = solver.solve()

# 11. Extract and plot results
X_traj = np.array(results["x_traj"])
t_grid = results["t_grid"]
lambda_res = np.array(results["lambda_list"])
c_res = np.array(results["c_res"])
U_traj = np.array(results["u_traj"])
t_grid_u = results["t_grid_u"]

plt.figure()
plt.plot(t_grid, c_res.squeeze(), label='c_res')
plt.title('c_res over time')
plt.xlabel('Time')  
plt.ylabel('c_res')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_grid[:-1], lambda_res.squeeze(), label='lambda_res')
plt.title('Lambda over time')
plt.xlabel('Time')
plt.ylabel('Lambda')
plt.legend()
plt.grid(True)      
plt.show()

plt.figure()
plt.plot(t_grid_u[:-1], U_traj, label='Control Trajectory')
plt.title('Control Trajectory over time')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)

# Unperturbed vector field
X, Y = np.meshgrid(np.arange(-6, 6, 1), np.arange(-6, 6, 1))
U = -0.2 * (X + 1)**2
V = -0.4 * (Y + 3)
ax.quiver(X, Y, U, V, color=np.array([27, 158, 119])/256, linewidth=1.5, label=r'$f(x)$')

# Constraint ellipse
a, b = 2, 4
x0, y0 = 0, 0
t = np.linspace(-np.pi, np.pi, 500)
x = x0 + a * np.cos(t)
y = y0 + b * np.sin(t)
ax.plot(x, y, '--r', linewidth=2, label=r'$c(x)$')

# Trajectory
ax.plot(X_traj[:, 0], X_traj[:, 1], '-b', linewidth=3, label=r'$x(t)$')

ax.set_xlim([-5.5, 5.5])
ax.set_ylim([-5.5, 5.5])
ax.set_xlabel(r'$x$', fontsize=fontsize)
ax.set_ylabel(r'$y$', fontsize=fontsize)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.show()

