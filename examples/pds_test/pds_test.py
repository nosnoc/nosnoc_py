import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from nosnoc.nosnoc_opts import NosnocOpts, DcsMode
from nosnoc.model import NosnocModel
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.solver import NosnocSolver
from nosnoc.helpers import NosnocSimLooper

# Create a minimal model for PDS.
# Define a state x in R^2 and an unconstrained dynamics function
n_x = 2  # number of state variables
x = ca.SX.sym('x', n_x)
x0 = [0.0, 1.14159]  # initial state

# For PDS, supply a list for f_unconstrained and c_pds
# f_unconstrained: Simple linear dynamics
f_unconstrained_expr = [ca.vertcat(x[1], -x[0])]  

# c_pds: Simple gap function
c_pds_expr = [ca.vertcat(x[1] +0.2)]  

# Parameters setup for PDS mode (all using time-varying parameters)
p_time_var = ca.SX.sym('p_time', 1)  
p_global = ca.SX.sym('p_global', 0)   
p_time_var_val = np.ones((1,1))
p_global_val = np.array([])



# Create model instance
model = NosnocModel(x=x, 
                   x0=x0, 
                   f_unconstrained=f_unconstrained_expr,
                   c_pds=c_pds_expr, 
                   p_time_var=p_time_var, 
                   p_global=p_global,
                   p_time_var_val=p_time_var_val, 
                   p_global_val=p_global_val)


# Construct options set to PDS mode.
opts = NosnocOpts(dcs_mode=DcsMode.PDS, n_s=3, terminal_time=10.0, use_fesd=True,sigma_0=1.0)
#opts.preprocess()



# Create a trivial OCP.
#ocp = NosnocOcp()
#ocp.preprocess_ocp(model)

# Build the problem.
#problem = NosnocProblem(opts, model, ocp)


# Create solver with additional debug info
solver = NosnocSolver(opts, model)
print("\nDEBUG: Solver Parameters")
print(f"Initial parameter vector shape: {solver.p0.shape if hasattr(solver, 'p0') else 'No p0'}")

# Solve the problem
#results = solver.solve()
looper = NosnocSimLooper(solver, model.x0, 31)
looper.run()
results = looper.get_results()
solver.problem.print()
# Extract the state trajectory and time grid
X_sim = np.array(results["X_sim"])  # shape: (N, n_x)
t_grid = results["t_grid"]
lambda_sim = np.array(results["lambda_sim"])  # shape: (N, n_x)

print("t_grid shape:", t_grid.shape)
print("lambda_sim shape:", lambda_sim.shape)
# Plot the state variables
plt.figure()
for i in range(X_sim.shape[1]):
    plt.plot(t_grid, X_sim[:, i], label=f'x[{i}]')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('PDS State Trajectory')
plt.legend()
plt.grid(True)
plt.show()

