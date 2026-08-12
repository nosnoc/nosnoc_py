from parametric_cart_pole_with_friction import get_default_opts, parameteric_cart_pole_model
from pendulum_utils import plot_results

import nosnoc as ns


def main():
    opts = get_default_opts(T=1.0, N_stages=10)
    model = parameteric_cart_pole_model(with_global_var=False)
    solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
    mpc_opts = ns.rtopt.FullMPCOptions(mpcc_solver_opts=solver_opts)
    mpc = ns.rtopt.FullMPC(model,opts,mpc_opts)
    #plot_results(solver)


if __name__ == "__main__":
    main()
