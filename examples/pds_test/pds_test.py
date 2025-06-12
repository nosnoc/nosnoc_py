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
opts = NosnocOpts(
    dcs_mode=DcsMode.PDS,
    n_s=3,
    terminal_time=10.0,
    use_fesd=True,
    sigma_0=1.0,
    comp_tol=1e-10,
    max_iter_homotopy=12,
    sigma_N=1e-11
    # ...add any other relevant options
)
#opts.preprocess()



# Create a trivial OCP.
#ocp = NosnocOcp()
#ocp.preprocess_ocp(model)

# Build the problem.
#problem = NosnocProblem(opts, model, ocp)


# Create solver with additional debug info
solver = NosnocSolver(opts, model)

# Solve the problem
#results = solver.solve()
looper = NosnocSimLooper(solver, model.x0, 10)
looper.run()
results = looper.get_results()
solver.problem.print()
# Extract the state trajectory and time grid
X_sim = np.array(results["X_sim"])  # shape: (N, n_x)
t_grid = results["t_grid"]
lambda_sim = np.array(results["lambda_sim"]) 
lambda_sim = np.squeeze(lambda_sim)           
lambda_plot = lambda_sim.flatten()            

print("t_grid shape:", t_grid.shape)
print("lambda_sim shape:", lambda_sim.shape)
print("Number of variables:", casadi_length(solver.problem.w))
print("Number of constraints:", casadi_length(solver.problem.g))
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

