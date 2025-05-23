import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from nosnoc.nosnoc_opts import NosnocOpts, DcsMode
from nosnoc.model import NosnocModel
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.solver import NosnocSolver

# Create a minimal model for PDS.
# Define a state x in R^2 and an unconstrained dynamics function.
n_x = 2  # number of state variables
x = ca.SX.sym('x', n_x)
x0 = np.zeros(n_x)

# For PDS, supply a list for f_unconstrained and c_pds.
# f_unconstrained: Here a simple linear dynamics.
f_unconstrained_expr = [ca.vertcat(-x[0] + 0.5*x[1], -x[1])]  # list with one expression (shape (2,1))

# c_pds: A simple gap function; here, for example, x[0] - 0.5.
c_pds_expr = [ca.vertcat(x[0] - 0.5)]  # list with one expression (shape (1,1))

# Other necessary inputs: minimal placeholders for p_time_var and p_global.
p_time_var = ca.SX.sym('p_time', 1)
p_global = ca.SX.sym('p_global', 1)
p_time_var_val = np.ones((1, 1))
p_global_val = np.array([1.0])

# The algebraic variable and constraint placeholders.
# (In a minimal example, we may set them to dummy values.)
z = ca.SX.sym('z', 0)
g_z = ca.SX([])

# Create the model instance in PDS mode by providing f_unconstrained and c_pds.
model = NosnocModel(x=x, x0=x0, f_unconstrained=f_unconstrained_expr,
                    c_pds=c_pds_expr, g_z=g_z, 
                    p_time_var=p_time_var, p_global=p_global,
                    p_time_var_val=p_time_var_val, p_global_val=p_global_val)


# Construct options set to PDS mode.
opts = NosnocOpts(dcs_mode=DcsMode.PDS, n_s=1, terminal_time=1.0, use_fesd=True)
opts.preprocess()

model.preprocess_model(opts)

# Create a trivial OCP.
ocp = NosnocOcp()
ocp.preprocess_ocp(model)

# Build the problem.
problem = NosnocProblem(opts, model, ocp)

# (Optionally) print some key elements to check that the minimal example is created correctly.
print("Minimal PDS example created successfully.")
print("Model state dimension:", model.x.shape)
print("Options dcs_mode:", opts.dcs_mode)

# Build the solver
solver = NosnocSolver(opts, model, ocp)

# Solve the problem
results = solver.solve()

# Extract the state trajectory and time grid
x_traj = np.array(results["x_traj"])  # shape: (N, n_x)
t_grid = results["t_grid"]

# Plot the state variables
plt.figure()
for i in range(x_traj.shape[1]):
    plt.plot(t_grid, x_traj[:, i], label=f'x[{i}]')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('PDS State Trajectory')
plt.legend()
plt.grid(True)
plt.show()