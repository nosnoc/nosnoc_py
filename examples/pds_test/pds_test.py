import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from nosnoc.nosnoc_opts import NosnocOpts, DcsMode
from nosnoc.model import NosnocModel
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.solver import NosnocSolver
from nosnoc.helpers import NosnocSimLooper
from nosnoc.utils import casadi_length, casadi_vertcat_list

# Create a minimal model for PDS.
# Define a state x in R^2 and an unconstrained dynamics function
n_x = 2  # number of state variables
x = ca.SX.sym('x', n_x)
x0 = [np.sqrt(2), np.sqrt(2)]  # initial state
  

# For PDS, supply a list for f_unconstrained and c_pds
# f_unconstrained: Simple linear dynamics
f_unconstrained_expr = [ca.vertcat(x[1], -x[0])]  

# c_pds: Simple gap function
c_pds_expr = [ca.vertcat(x[1] + 1)]  



# Create model instance
model = NosnocModel(x=x, 
                   x0=x0, 
                   f_unconstrained=f_unconstrained_expr,
                   c_pds=c_pds_expr)


# Construct options set to PDS mode.
opts = NosnocOpts(
    dcs_mode=DcsMode.PDS,
    n_s=3,
    terminal_time= 0.1 ,
    use_fesd=True,
    sigma_0=1e-4,
    comp_tol=1e-10,
    N_finite_elements=2,
    sigma_N=1e-11,
    print_level=2
)
#opts.preprocess()



# Create a trivial OCP.
#ocp = NosnocOcp()
#ocp.preprocess_ocp(model)

# Build the problem.
#problem = NosnocProblem(opts, model)


# Create solver with additional debug info
solver = NosnocSolver(opts, model)

# Solve the problem
#results = solver.solve(
looper = NosnocSimLooper(solver, model.x0, Nsim = 46)
looper.run()
results = looper.get_results()
#solver.problem.print()
# Extract the state trajectory and time grid
X_sim = np.array(results["X_sim"])  # shape: (N, n_x)
t_grid = results["t_grid"]
lambda_sim = np.array(results["lambda_sim"]) 
lambda_sim = np.squeeze(lambda_sim)           
lambda_plot = lambda_sim.flatten()            


# Plot the state variables
plt.figure()
for i in range(X_sim.shape[1]):
    plt.plot(t_grid, X_sim[:, i], label=f'x[{i}]')
plt.plot(t_grid[:-1], lambda_plot, label=r'$\lambda(t)$', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PDS State and Lambda Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# Plot x[0] vs x[1] (phase plot)
plt.figure()
plt.plot(X_sim[:, 0], X_sim[:, 1], label='Phase Plot', color='blue')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Phase Plot: x[0] vs x[1]')
plt.grid(True)
plt.legend()
plt.show()

# Get the final simulated state
x_final = X_sim[-1, :]  # shape: (2,) 
print(f"Final state x: {x_final}")
# Reference point
x_ref = [-1, 0]

# Compute Euclidean distance
distance = np.linalg.norm(x_final - x_ref)

print(f"Euclidean distance between final x and (-1, 0): {distance:.6f}")

