from parametric_cart_pole_with_friction import get_default_opts, parameteric_cart_pole_model
from pendulum_utils import _plot_results

import nosnoc as ns
import numpy as np


def main():
    opts = get_default_opts(T=1.0, N_stages=10)
    model = parameteric_cart_pole_model(with_global_var=False)
    solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    mpc_opts = ns.rtopt.FullMPCOptions(mpcc_solver_opts=solver_opts)
    mpc = ns.rtopt.FullMPC(model,opts,mpc_opts)

    dt = opts.T/opts.N_stages

    X = [model.x0]
    T = [0.0]
    U = []
    N_mpc = 50
    for ii in range(N_mpc):
        mpc.prepare(x_pred=X[-1])
        u = mpc.optimize(x0=X[-1])
        x_pred = mpc.get_predicted_state()
        U.append(u)
        X.append(x_pred)
        T.append(T[-1] + dt)

    _plot_results(np.vstack(X),np.array(U),np.array(T),np.array(T))


if __name__ == "__main__":
    main()
