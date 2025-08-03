import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from nosnoc.nosnoc_opts import NosnocOpts, DcsMode
from nosnoc.model import NosnocModel
from nosnoc.nosnoc_types import CrossComplementarityMode
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.solver import NosnocSolver
from nosnoc.helpers import NosnocSimLooper

T =(11*np.pi/12) + np.sqrt(3)

def solve_and_plot(model, opts, title):
    # Preprocess model and create problem
    model.preprocess_model(opts)
    #ocp = NosnocOcp()
    #ocp.preprocess_ocp(model)
    #problem = NosnocProblem(opts, model, ocp)
    
    # Build and run solver
    
    solver = NosnocSolver(opts, model)
    looper = NosnocSimLooper(solver, x0=model.x0,Tsim= T)
    looper.run()
    results = looper.get_results()

    
    # Plot results
    x_traj = np.array(results["X_sim"])
    t_grid = results["t_grid"]
    lambda_sim = np.array(results["lambda_sim"])
    return x_traj, t_grid, lambda_sim
    
    

# Define system dimensions and variables
n_x = 1
x = ca.SX.sym('x', n_x)
x0 = np.array([1.0])  # Initial condition for x
f11 = 3
f12 = 1  

F = [ca.horzcat(f11,f12)]

# Selection matrices for the two modes
S = [np.array([[-1], [1]])]

# Switching function
c = [x]  # Switches at x1 = 0

g_stewart = [np.array([[-x], [x]])]  # Stewart representation requires g_Stewart

# Algebraic variables


# Create models for different reformulations
model_stewart = NosnocModel(
    x=x, x0=x0,
    F=F, S=S, c=c,
    g_Stewart=g_stewart  
)


# Create and solve problems for each reformulation
# Stewart
opts_stewart = NosnocOpts(
    dcs_mode=DcsMode.STEWART,
    n_s=2,
    terminal_time=T/100,
    use_fesd=True,
    cross_comp_mode=CrossComplementarityMode.COMPLEMENT_ALL_STAGE_VALUES_WITH_EACH_OTHER,
    print_level=2,
)

opts_stewart.preprocess()
x_traj_stewart, t_grid_stewart, lambda_sim = solve_and_plot(model_stewart, opts_stewart, "Stewart")

# Plot Stewart results
plt.figure()
for i in range(x_traj_stewart.shape[1]):
    plt.plot(t_grid_stewart, x_traj_stewart[:, i], label=f'x[{i}]')
lambda_plot = np.squeeze(lambda_sim)  # shape: (100, 2)
t_lambda = t_grid_stewart[:lambda_plot.shape[0]]
for i in range(lambda_plot.shape[1]):
    plt.plot(t_lambda, lambda_plot[:, i], label=f'$\lambda_{{{i}}}(t)$')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('PDS State Trajectory')
plt.legend()
plt.grid(True)
plt.show()


# Step


