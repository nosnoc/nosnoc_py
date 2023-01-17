import nosnoc
from casadi import MX
from math import ceil, log
import numpy as np


def create_options():
    opts = nosnoc.NosnocOpts()
    # Degree of interpolating polynomial
    opts.n_s = 2
    # === MPCC settings ===
    # upper bound for elastic variables
    opts.s_elastic_max = 1e1
    # in penalty methods  1: J = J+(1/p)*J_comp (direct)  , 0 : J = p*J+J_comp (inverse)
    opts.objective_scaling_direct = 0
    # === Penalty/Relaxation paraemetr ===
    # starting smouothing parameter
    opts.sigma_0 = 1e1
    # end smoothing parameter
    opts.sigma_N = 1e-10
    # decrease rate
    opts.homotopy_update_slope = 0.1
    # number of steps
    opts.N_homotopy = ceil(
        abs(log(opts.sigma_N/opts.sigma_0)/log(opts.homotopy_update_slope)))+1
    opts.comp_tol = 1e-14

    # IPOPT Settings
    opts.opts_casadi_nlp['ipopt']['max_iter'] = 250

    # New setting: time freezing settings
    opts.initial_theta = 0.5
    return opts


def create_temp_control_model_voronoi():
    # Discretization parameters
    N_stages = 2
    N_finite_elements = 1
    T = 0.1  # (here determined latter depeding on omega)
    h = T/N_stages

    # inital value
    t0 = 0
    w0 = 0
    y0 = 15

    lambda_cool_down = -0.2  # cool down time constant of lin dynamics
    u_heat = 10  # heater power

    # jump points in x in the hysteresis function
    y1 = 18
    y2 = 20

    z1 = np.array([1/4, -1/4])
    z2 = np.array([1/4, 1/4])
    z3 = np.array([3/4, 3/4])
    z4 = np.array([3/4, 5/4])
    # Z = [1/4 1/4 3/4 3/4;...
    #      -1/4 1/4 3/4 5/4]
    Z = np.concatenate([z1, z2, z3, z4])

    # Inital Value
    x0 = np.array([y0, w0, t0]).T

    # Define model dimensions, equations, constraint functions, regions an so on.
    # number of Cartesian products in the model ("independet switches"), we call this layer
    n_sys = 1
    m_1 = 4
    n_f_sys = [m_1]
    # Variable defintion
    y = MX.sym('y')
    w = MX.sym('w')
    t = MX.sym('t')

    x = [y, w, t]
    n_x = length(x)
    lbx = -inf*ones(n_x, 1)
    ubx = inf*ones(n_x, 1)

    # linear transformation for rescaling of the switching function.
    psi = (y-y1)/(y2-y1)
    z = np.array([psi, w])

    # discriminant functions via voronoi
    g_11 = np.linalg.norm(z-z1) ^ 2
    g_12 = np.linalg.norm(z-z2) ^ 2
    g_13 = np.linalg.norm(z-z3) ^ 2
    g_14 = np.linalg.norm(z-z4) ^ 2

    g_ind = np.array([[g_11], [g_12], [g_13], [g_14]])
    c = g_ind

    # control
    u = MX.sym('u')
    s = MX.sym('s')  # Length of time
    n_u = 1
    u0 = [0]
    umax = 1e-3

    lbu = -umax*np.ones((n_u, 1))
    ubu = umax*np.ones((n_u, 1))

    # System dynamics:
    # Heating:
    f_A = np.array([lambda_cool_down*y+u, 0, 1]).T
    f_B = np.array([lambda_cool_down*y, 0, 1]).T

    a_push = 5
    f_push_down = np.array([0, a_push*(psi-1) ^ 2/(1+(psi-1) ^ 2), 0])
    f_push_up = np.array([0, a_push*(psi) ^ 2/(1+(psi) ^ 2), 0])

    f_11 = 2*f_A-f_push_down
    f_12 = f_push_down
    f_13 = f_push_up
    f_14 = 2*f_B-f_push_up
    f_1 = np.concatenate([f_11, f_12, f_13, f_14])
    F = f_1

    # objective
    f_q = (u ^ 2)+y ^ 2


# # Generate Model
# model = temp_control_model_voronoi()
# # - Simulation settings
# model.T_sim = 3
# model.N_stages = 1
# model.N_finite_elements = 2
# model.N_sim = 30
# settings.use_previous_solution_as_initial_guess = 1
# # Call FESD Integrator
# [results, stats] = integrator_fesd(model, settings)
