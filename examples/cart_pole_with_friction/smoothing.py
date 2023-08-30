import casadi as ca
import numpy as np
from nosnoc import casadi_length, NosnocOcp, NosnocAutoModel, print_casadi_vector, generate_butcher_tableu_integral, IrkSchemes
from cart_pole_with_friction import get_cart_pole_model_and_ocp
from pendulum_utils import plot_results, plot_sigma_experiment


def setup_collocation_nlp(model: NosnocAutoModel, ocp: NosnocOcp, T: float, N: int, n_s: int,
                          N_FE: int):
    B, C, D, _ = generate_butcher_tableu_integral(n_s, IrkSchemes.RADAU_IIA)

    # Time horizon
    T = 10.

    x = model.x
    x0 = model.x0
    u = model.u
    nx = casadi_length(x)
    nu = casadi_length(u)

    # Continuous time dynamics
    f_ode = ca.Function('f', [x, u, model.p], [model.f_nonsmooth_ode, ocp.f_q])
    f_terminal = ca.Function('f_terminal', [x], [ocp.f_terminal])

    # Control discretization
    h = T / (N * N_FE)

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    objective = 0
    g = []
    lbg = []
    ubg = []

    # "Lift" initial conditions
    Xk = ca.SX.sym('X0', nx)
    w.append(Xk)
    lbw.append(x0)
    ubw.append(x0)
    w0.append(x0)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.SX.sym('U_' + str(k))
        w.append(Uk)
        lbw.append(ocp.lbu)
        ubw.append(ocp.ubu)
        w0.append(np.zeros((nu,)))

        # Loop over integration steps / finite elements
        for i_fe in range(N_FE):
            Xk_end = D[0] * Xk
            # State at collocation points
            Xc = []
            for j in range(n_s):
                Xkj = ca.SX.sym(f'X_{k}_{i_fe}_{j}', nx)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(ocp.lbx)
                ubw.append(ocp.ubx)
                w0.append(np.zeros((nx,)))

            # Loop over collocation points
            for j in range(1, n_s + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(n_s):
                    xp = xp + C[r + 1, j] * Xc[r]

                # Append collocation equations
                fj, qj = f_ode(Xc[j - 1], Uk, model.p)
                g.append(h * fj - xp)
                lbg.append(np.zeros((nx,)))
                ubg.append(np.zeros((nx,)))

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                objective = objective + B[j] * qj * h

            # New NLP variable for state at end of interval
            Xk = ca.SX.sym(f'X_{k+1}', nx)
            w.append(Xk)
            lbw.append(ocp.lbx)
            ubw.append(ocp.ubx)
            w0.append(np.zeros((nx,)))

            # Add equality constraint
            g.append(Xk_end - Xk)
            lbg.append(np.zeros((nx,)))
            ubg.append(np.zeros((nx,)))

    objective += f_terminal(Xk)

    # Concatenate vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # NLP dict
    nlp = {
        'f': objective,
        'x': w,
        'g': g,
        'p': model.p,
        'w0': w0,
        'lbw': lbw,
        'ubw': ubw,
        'lbg': lbg,
        'ubg': ubg
    }
    casadi_nlp = {'f': objective, 'x': w, 'g': g, 'p': model.p}
    return nlp, casadi_nlp


def collocation_ocp_example():
    terminal_time = 5.0
    N = 30
    n_s = 1
    N_FE = 2
    x_ref = np.array([0, 180 / 180 * np.pi, 0, 0])  # end upwards

    # smoothing parameter
    sigma0 = 1e0

    model, ocp = get_cart_pole_model_and_ocp(F_friction=2.0, use_fillipov=False)
    nlp, casadi_nlp = setup_collocation_nlp(model, ocp, terminal_time, N, n_s, N_FE)

    solver = ca.nlpsol('solver', 'ipopt', casadi_nlp)

    sol = solver(x0=nlp['w0'],
                 lbx=nlp['lbw'],
                 ubx=nlp['ubw'],
                 lbg=nlp['lbg'],
                 ubg=nlp['ubg'],
                 p=sigma0)

    w_opt = sol['x'].full()

    # split into x and u values
    nx = casadi_length(model.x)
    nu = casadi_length(model.u)
    idx_diff = nu + nx * (n_s * N_FE) + (N_FE) * nx  # between x values at shooting nodes
    x_traj = np.hstack([w_opt[i::idx_diff] for i in range(nx)])
    u_traj = w_opt[nx::idx_diff].tolist()
    t_grid = np.linspace(0, terminal_time, N + 1)

    distance_to_target = np.abs(x_ref-x_traj[-1,:])
    print(f"{distance_to_target=}")

    results = {'x_traj': x_traj, 'u_traj': u_traj, 't_grid': t_grid, 't_grid_u': t_grid}
    plot_results(results)


def simulate_smoothed_model_cart_pole():
    terminal_time = 5.0
    Nsim = 100
    model, ocp = get_cart_pole_model_and_ocp(F_friction=2.0, use_fillipov=False)

    # create integrator
    p_integrator = ca.vertcat(model.u, model.p)
    Phi = ca.integrator('integrator', 'idas', {
        'x': model.x,
        'ode': model.f_nonsmooth_ode,
        'p': p_integrator
    }, {'tf': terminal_time / Nsim})

    nx = casadi_length(model.x)

    # define parameters / controls
    p_traj = np.ones((casadi_length(p_integrator), Nsim))
    p_traj[0, Nsim // 2:] = -2

    xcurrent = np.zeros((nx,))
    simX = np.zeros((nx, Nsim + 1))
    simX[:, 0] = xcurrent
    # simulation loop
    for i in range(Nsim):
        out = Phi(x0=xcurrent, p=p_traj[:, i])
        xcurrent = out['xf'].full().flatten()
        simX[:, i + 1] = xcurrent

    t_grid = np.linspace(0, terminal_time, Nsim + 1)
    results = {
        'x_traj': simX.T,
        't_grid': t_grid,
        't_grid_u': t_grid,
        'u_traj': p_traj[0, :].tolist()
    }
    plot_results(results)


def smoothing_collocation_experiment():
    terminal_time = 5.0
    N = 30
    n_s = 1
    N_FE = 2

    model, ocp = get_cart_pole_model_and_ocp(F_friction=2.0, use_fillipov=False)
    nlp, casadi_nlp = setup_collocation_nlp(model, ocp, terminal_time, N, n_s, N_FE)

    solver = ca.nlpsol('solver', 'ipopt', casadi_nlp)

    # smoothing experiment
    n_sigmas = 15
    sigma_values = np.logspace(1, -8, n_sigmas)
    objective_values = np.zeros((n_sigmas,))
    for i, sigma0 in enumerate(sigma_values):
        sol = solver(x0=nlp['w0'],
                     lbx=nlp['lbw'],
                     ubx=nlp['ubw'],
                     lbg=nlp['lbg'],
                     ubg=nlp['ubg'],
                     p=sigma0)
        objective_values[i] = sol['f'].full().flatten()

    # print(f"{sigma_values=} {objective_values=}")
    plot_sigma_experiment(sigma_values, objective_values)


if __name__ == "__main__":
    # simulate_smoothed_model_cart_pole()
    collocation_ocp_example()
    # smoothing_collocation_experiment()
