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

def create_generic_mpcc2():
    x = ca.SX.sym('x',8);
    x_0 = x[0:4]
    x_1 = x[4:6]
    x_2 = x[6:8]

    rho = 2
    lambda1= 3.9375
    lambda2 = -6.5
    lambda3 = -0.25
    lambda4 = 2.5

    f = 0.5*((x_0[0]-x_0[2])**2+(x_0[1]-x_0[3])**2) + lambda1*(-34+2*x_0[2]+8/3*x_0[3]+x_2[0]) - lambda2*(-24.25+1.25*x_0[2]+2*x_0[3]+x_2[1]) - lambda3*(x_1[0]+x_0[1]+x_0[2]-15) + lambda4*(x_1[1]+x_0[0]-x_0[3]-15) + 0.5*rho*( (-34+2*x_0[2]+8./3*x_0[3]+x_2[0])**2 + (-24.25+1.25*x_0[2]+2*x_0[3]+x_2[1])**2 + (x_1[0]+x_0[1]+x_0[2]-15)**2+ (x_1[1]+x_0[0]-x_0[3]-15)**2 )

    x0 = np.zeros(8)
    lbx = -np.inf*np.ones(8)
    ubx = np.inf*np.ones(8)

    mpcc = {
        "x": x,
        "g": ca.SX([]),
        "p": ca.SX([]),
        "G": x_1,
        "H": 2*x_2,
        "f": f,
    }

    solver_initialization = {
        "x0": x0,
        "lbx": lbx,
        "ubx": ubx,
    }
    return mpcc, solver_initialization

if __name__ == "__main__":
    solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions(assume_lower_bounds=False)

    mpcc1, init1 = create_generic_mpcc1()
    solver1 = ns.mpccsol.mpccsol("reg_homotopy", mpcc1, solver_opts)
    solution1 = solver1(**init1)

    mpcc2, init2 = create_generic_mpcc2()
    solver2 = ns.mpccsol.mpccsol("reg_homotopy", mpcc2, solver_opts)
    solution2 = solver2(**init2)
    breakpoint()
