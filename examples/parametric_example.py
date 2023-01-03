import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

from simplest_example import get_default_options, plot_results
import nosnoc

TOL = 1e-9

# Analytic solution
EXACT_SWITCH_TIME = 1 / 3
TSIM = np.pi / 4

# Initial Value
X0 = np.array([-1.0])


def get_simplest_parametric_model_switch():
    # Variable defintion
    x1 = SX.sym("x1")
    x = x1
    # every constraint function corresponds to a sys (note that the c_i might be vector valued)
    c = [x1]
    # sign matrix for the modes
    S = [np.array([[-1], [1]])]

    p = SX.sym('p')
    p_val = np.array([-1.0])

    f_11 = 3
    f_12 = p

    # in matrix form
    F = [horzcat(f_11, f_12)]

    model = nosnoc.NosnocModel(x=x, F=F, S=S, c=c, x0=X0, name='simplest_switch', p=p, p_val = p_val)

    return model


# EXAMPLE
def example():
    model = get_simplest_parametric_model_switch()

    opts = get_default_options()
    opts.print_level = 1
    Nsim = 1
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, X0, Nsim)
    looper.run()
    results = looper.get_results()

    plot_results(results)


if __name__ == "__main__":
    example()
