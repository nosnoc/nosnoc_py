from parametric_cart_pole_with_friction import get_default_opts, parameteric_cart_pole_model
from pendulum_utils import _plot_results

from casadi import SX, horzcat, vertcat, cos, sin, inv
import nosnoc as ns
import numpy as np

CCOPT = False
RTI = True

T_OCP = 1.0
N_STAGES = 10
T_STEP = T_OCP/N_STAGES
T_MPC = 5.0
N_MPC = round(T_MPC/T_STEP)
N_SIM = 5

def cartpole_mpc_model():
    ## Model defintion
    q = SX.sym('q', 2)
    v = SX.sym('v', 2)
    x = vertcat(q, v)
    u = SX.sym('u')  # control

    ## parametric version:
    # masses
    m1 = SX.sym('m1')  # cart
    m2 = SX.sym('m2')  # link
    x_ref = SX.sym('x_ref', 4)
    u_ref = SX.sym('u_ref', 1)
    x_ref_val = np.array([0, 180 / 180 * np.pi, 0, 0])  # end upwards
    u_ref_val = np.array([0.0])

    p_global = vertcat(x_ref, u_ref, m1, m2)

    p_global_val = np.concatenate([x_ref_val,u_ref_val,np.array([1.0, 0.1])])

    link_length = 1
    g = 9.81
    # Inertia matrix
    M = vertcat(horzcat(m1 + m2, m2 * link_length * cos(q[1])),
                horzcat(m2 * link_length * cos(q[1]), m2 * link_length**2))
    # Coriolis force
    C = SX.zeros(2, 2)
    C[0, 1] = -m2 * link_length * v[1] * sin(q[1])

    # all forces = Gravity+Control+Coriolis (+Friction)
    f_all = vertcat(u, -m2 * g * link_length * sin(x[1])) - C @ v

    # friction between cart and ground
    F_friction = 2
    # Dynamics with $ v > 0$
    f_1 = vertcat(v, inv(M) @ (f_all - vertcat(F_friction, 0)))
    # Dynamics with $ v < 0$
    f_2 = vertcat(v, inv(M) @ (f_all + vertcat(F_friction, 0)))

    F = [horzcat(f_1, f_2)]
    # switching function (cart velocity)
    c = [v[0]]
    # Sign matrix # f_1 for c=v>0, f_2 for c=v<0
    S = [np.array([[1], [-1]])]

    # specify initial and end state, cost ref and weight matrix
    x0 = np.array([1, 0 / 180 * np.pi, 0, 0])  # start downwards
    #x0 = np.array([0.0, 0 / 180 * np.pi, 0, 0])  # start downwards

    Q = np.diag([1, 10, 1, 1])
    Q_terminal = np.diag([1000, 1000, 1, 1])
    R = 0.1

    # Stage cost
    f_q = 0.5*((x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref))
    # terminal cost
    f_terminal = 0.5*((x - x_ref).T @ Q_terminal @ (x - x_ref))

    # bounds
    ubx = np.array([5.0, np.inf, np.inf, np.inf])
    lbx = -np.array([5.0, np.inf, np.inf, np.inf])

    u_max = 20.0
    lbu = -np.array([u_max])
    ubu = np.array([u_max])

    model = ns.model.Pss(
        x=x,
        F=F,
        S=S,
        c=c,
        x0=x0,
        u=u,
        p_global=p_global,
        p_global_val=p_global_val,
        lbu=lbu,
        ubu=ubu,
        f_q=f_q,
        f_q_T=f_terminal,
        lbx=lbx,
        ubx=ubx,
    )
    return model

def cartpole_dynamics_model():
    ## Model defintion
    q = SX.sym('q', 2)
    v = SX.sym('v', 2)
    x = vertcat(q, v)
    u = SX.sym('u')  # control

    ## parametric version:
    # masses
    m1 = SX.sym('m1')  # cart
    m2 = SX.sym('m2')  # link
    p_global = vertcat(m1, m2)
    p_global_val = np.array([1.0, 0.1])
    link_length = 1
    g = 9.81
    # Inertia matrix
    M = vertcat(horzcat(m1 + m2, m2 * link_length * cos(q[1])),
                horzcat(m2 * link_length * cos(q[1]), m2 * link_length**2))
    # Coriolis force
    C = SX.zeros(2, 2)
    C[0, 1] = -m2 * link_length * v[1] * sin(q[1])

    # all forces = Gravity+Control+Coriolis (+Friction)
    f_all = vertcat(u, -m2 * g * link_length * sin(x[1])) - C @ v

    # friction between cart and ground
    F_friction = 2
    # Dynamics with $ v > 0$
    f_1 = vertcat(v, inv(M) @ (f_all - vertcat(F_friction, 0)))
    # Dynamics with $ v < 0$
    f_2 = vertcat(v, inv(M) @ (f_all + vertcat(F_friction, 0)))

    F = [horzcat(f_1, f_2)]
    # switching function (cart velocity)
    c = [v[0]]
    # Sign matrix # f_1 for c=v>0, f_2 for c=v<0
    S = [np.array([[1], [-1]])]

    # specify initial and end state, cost ref and weight matrix
    #x0 = np.array([1, 0 / 180 * np.pi, 0, 0])  # start downwards
    x0 = np.array([0.0, 0 / 180 * np.pi, 0, 0])  # start downwards

    model = ns.model.Pss(
        x=x,
        F=F,
        S=S,
        c=c,
        x0=x0,
        u=u,
        p_global=p_global,
        p_global_val=p_global_val,
    )
    return model

def _build_full_mpc():
    opts = get_default_opts(T=T_OCP, N_stages=N_STAGES, cross_comp_mode=ns.CrossComplementarityMode.FE_FE)
    model = cartpole_mpc_model()
    if CCOPT:
        solver_opts = ns.mpccsol.plugins.ccopt.CCOptOptions()
        solver_opts.madnlp_opts["tol"] = 1e-6
        solver_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        #solver_opts.madnlp_opts["barrier.TYPE"] = "MonotoneUpdate"
        #solver_opts.ccopt_opts["relaxation_update.TYPE"] = "RolloffRelaxationUpdate"
        #solver_opts.ccopt_opts["relaxation_update.sigma_min"] = 1e-7
    else:
        solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    mpc_opts = ns.rtopt.FullMPCOptions(mpcc_solver_opts=solver_opts)
    mpc = ns.rtopt.FullMPC(model,opts,mpc_opts)

    return opts,model,mpc

def _build_rti():
    opts = get_default_opts(T=T_OCP, N_stages=N_STAGES, n_s=3, cross_comp_mode=ns.CrossComplementarityMode.FE_FE)
    model = cartpole_mpc_model()
    if CCOPT:
        mpcc_opts = ns.mpccsol.plugins.ccopt.CCOptOptions()
        mpcc_opts.madnlp_opts["tol"] = 1e-8
        mpcc_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        mpcc_opts.madnlp_opts["print_level"] = 5
        qpcc_opts = ns.mpccsol.plugins.ccopt.CCOptOptions()
        qpcc_opts.madnlp_opts["tol"] = 1e-6
        qpcc_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        qpcc_opts.madnlp_opts["print_level"] = 6
        qpcc_opts.madnlp_opts["disable_garbage_collector"] = True
        #solver_opts.madnlp_opts["barrier.TYPE"] = "MonotoneUpdate"
        qpcc_opts.ccopt_opts["relaxation_update.TYPE"] = "RolloffRelaxationUpdate"
        qpcc_opts.ccopt_opts["print_level"] = 6
        #solver_opts.ccopt_opts["relaxation_update.sigma_min"] = 1e-7
    else:
        mpcc_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        qpcc_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        #qpcc_opts.opts_casadi_nlp["ipopt"]["print_level"] = 5
    mpc_opts = ns.rtopt.RTIMPCOptions(
        mpcc_solver_opts=mpcc_opts,
        qpcc_solver_opts=qpcc_opts,
        prepare_step=ns.rtopt.PreparationStep.FULL,
        n_advanced_steps=3,
    )
    mpc = ns.rtopt.RTIMPC(model,opts,mpc_opts)

    return opts,model,mpc

def _build_integrator():
    opts = get_default_opts(T=T_OCP/N_STAGES, N_stages=1, n_s=4, cross_comp_mode=ns.CrossComplementarityMode.FE_FE)
    model = cartpole_dynamics_model()
    mpcc_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions(
        homotopy_steering_strategy = ns.mpccsol.plugins.reg_homotopy.HomotopySteeringStrategy.ELL_INF
    )
    integrator_opts = ns.FESDIntegratorOptions(solver_opts=mpcc_opts, T_sim=T_STEP, N_sim=N_SIM, print_level=0)

    return ns.Integrator(model, opts, integrator_opts)

def main():

    if RTI:
        opts,model,mpc = _build_rti()
    else:
        opts,model,mpc = _build_full_mpc()
    integrator = _build_integrator()

    x_last = model.x0
    X = [model.x0]
    t_grid = [[0.0]]
    U = []
    control_grid = [0.0]
    for ii in range(N_MPC):
        u = mpc.optimize(x0=x_last)
        u_sim = np.kron(np.ones((N_SIM,1)), u)
        t_grid_ii, x_ii,_,_ = integrator.simulate(x_last, u=u_sim)
        X.append(x_ii[1:,:])
        U.append(u)
        t_grid.append(control_grid[-1]+t_grid_ii[1:])
        control_grid.append(control_grid[-1] + T_STEP)
        x_last = x_ii[-1,:]
        #x_last = mpc.get_predicted_state()
        print(f"x_last = {x_last}")
        print(f"x_pred = {mpc.get_predicted_state()}")
        print(f"error = {np.linalg.norm(x_last - mpc.get_predicted_state())}")
        print(f"u = {u}")
        mpc.prepare(x_pred=mpc.get_predicted_state())

    _plot_results(np.vstack(X),np.array(U),np.concatenate(t_grid),np.array(control_grid))


if __name__ == "__main__":
    main()
