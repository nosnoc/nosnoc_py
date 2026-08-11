import casadi as ca
import nosnoc as ns
import numpy  as np


def create_generic_mpcc1():
    # Variables
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    # Parameters
    p = ca.SX.sym('p')
    x = ca.vertcat(x1,x2)
    f = (x1-1)**2+x2**3+x2**2+p
    x0 = np.zeros(2)
    p0 = np.zeros(1)
    lbx = np.zeros(1)
    ubx = np.inf*np.ones(2)

    mpcc = {
        "x": x,
        "g": x1+x2,
        "f": f,
        "p": p,
        "G": x1,
        "H": x2,
    }
    solver_initialization = {
        "lbx": lbx,
        "ubx": ubx,
        "lbg": -np.inf*np.ones(1),
        "ubg": 10.0*np.ones(1),
    }
    return mpcc, solver_initialization


if __name__ == "__main__":
    mpcc, init = create_generic_mpcc1()
    solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()

    solver1 = ns.mpccsol.mpccsol("reg_homotopy", mpcc, solver_opts)

    solution1 = solver(**init)
