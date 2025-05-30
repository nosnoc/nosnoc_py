import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from nosnoc.nosnoc_opts import NosnocOpts, DcsMode
from nosnoc.model import NosnocModel
from nosnoc.ocp import NosnocOcp
from nosnoc.problem import NosnocProblem
from nosnoc.solver import NosnocSolver

def solve_and_plot(model, opts, title):
    # Preprocess model and create problem
    model.preprocess_model(opts)
    ocp = NosnocOcp()
    ocp.preprocess_ocp(model)
    problem = NosnocProblem(opts, model, ocp)
    
    # Build and run solver
    solver = NosnocSolver(opts, model, ocp)
    results = solver.solve()
    
    # Plot results
    x_traj = np.array(results["x_traj"])
    t_grid = results["t_grid"]
    
    plt.figure()
    for i in range(x_traj.shape[1]):
        plt.plot(t_grid, x_traj[:, i], label=f'x[{i}]')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'{title} State Trajectory')
    plt.legend()
    plt.grid(True)
    return x_traj, t_grid

# Define system dimensions and variables
n_x = 2
n_z = 0
n_theta = 2
x = ca.SX.sym('x', n_x)
z = ca.SX.sym('z', n_z)
x0 = np.array([1.0, 0.0])  # Initial condition for x

# Define a relay system with two modes
# Mode 1: dx1/dt = x2, dx2/dt = -1
# Mode 2: dx1/dt = x2, dx2/dt = +1
# Switching function: x1 (switches at x1 = 0)
F = [
    ca.vertcat(x[1], -1.0),  # Mode 1
    ca.vertcat(x[1], 1.0)    # Mode 2
]

# Selection matrices for the two modes
S = [ca.DM([[1.0]])]*2

# Switching function
c = [x[0],-x[0]]  # Switches at x1 = 0

# Parameters
p_time_var = ca.SX.sym('p_time', 1)
p_global = ca.SX.sym('p_global', 1)
p_time_var_val = np.ones((1, 1))
p_global_val = np.array([1.0])

# Algebraic variables


# Create models for different reformulations
model_stewart = NosnocModel(
    x=x, x0=x0, z=z,
    F=F, S=S, c=c,
    g_Stewart=c,  # Stewart requires g_Stewart
    p_time_var=p_time_var, p_global=p_global,
    p_time_var_val=p_time_var_val, p_global_val=p_global_val
)

model_step = NosnocModel(
    x=x, x0=x0,
    F=F, S=S, c=c,
    p_time_var=p_time_var, p_global=p_global,
    p_time_var_val=p_time_var_val, p_global_val=p_global_val
)

# Create and solve problems for each reformulation
# Stewart
#opts_stewart = NosnocOpts(
#    dcs_mode=DcsMode.STEWART,
#    n_s=2,
#    terminal_time=1.0,
#    use_fesd=True
#)

#opts_stewart.preprocess()
#x_traj_stewart, t_grid_stewart = solve_and_plot(model_stewart, opts_stewart, "Stewart")

# Step
opts_step = NosnocOpts(
    dcs_mode=DcsMode.STEP,
    n_s=2,
    terminal_time=1.0,
    use_fesd=True
)
opts_step.preprocess()
x_traj_step, t_grid_step = solve_and_plot(model_step, opts_step, "Step")

plt.show()