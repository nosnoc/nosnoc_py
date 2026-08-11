from copy import copy
from dataclasses import dataclass

import casadi as ca
import numpy as np
from .mpccsol import mpccsol
from .mpccsol.plugins.reg_homotopy import RegHomotopyOptions

@dataclass
class QpccDims():
    nx: int
    ng: int
    ncc: int
    
class Qpcc():
    """
    Build a parametric QPCC with the given sparsity pattern.
    Currently assumes that the original MPCC is in the form of:
        min f(x,p)
         x
         s.t. lbx <= x <= ubx
              lbg <= g(x,p) <= ubg
              0 <= G(x,p) _|_ H(x,p) >= 0
    and is passed as a nosnoc.MPCC vdx object
    """
    def __init__(self, mpcc, use_mpcc_multipliers=False):
        """
        Construct the required functions required to form a QPCC from the passed `mpcc`.
        This includes constructing the Lagrange-Hessian function, constraint Jacobian, and objective gradient.
        Space is allocated for evaluating these functions.
        """
        self.mpcc = copy(mpcc)
        self.use_mpcc_multipliers = mpcc
        dims = QpccDims(nx=len(mpcc.w), ng=len(mpcc.g), ncc=len(mpcc.G))
        self.dims = dims
        ## linearize around mpcc.w.init
        # Build L
        lam_g = mpcc.g.symbolic_type.sym("lam_g", dims.ng)
        lam_G = mpcc.G.symbolic_type.sym("lam_G", dims.ncc)
        lam_H = mpcc.H.symbolic_type.sym("lam_H", dims.ncc)
        if use_mpcc_multipliers: # TODO(@anton) check sign convention for lam_G, lam_H
            L = mpcc.f - ca.dot(lam_g, mpcc.g.sym) - ca.dot(lam_G, mpcc.G.sym) - ca.dot(lam_H, mpcc.H.sym)
        else:
            L = mpcc.f - ca.dot(lam_g, mpcc.g.sym)
        hess_L,nabla_L = ca.hessian(L, mpcc.w.sym)
        grad_f = ca.gradient(mpcc.f, mpcc.w.sym)
        jac_g = ca.jacobian(mpcc.g.sym, mpcc.w.sym)
        jac_G = ca.jacobian(mpcc.G.sym, mpcc.w.sym)
        jac_H = ca.jacobian(mpcc.H.sym, mpcc.w.sym)

        self.Q_sparsity = hess_L.sparsity()
        self.A_sparsity = jac_g.sparsity()
        self.G_sparsity = jac_G.sparsity()
        self.H_sparsity = jac_H.sparsity()

        self.Q_fun = ca.Function("Q", [mpcc.w.sym, lam_g, lam_G, lam_H, mpcc.p.sym], [hess_L])
        self.q_fun = ca.Function("q", [mpcc.w.sym,mpcc.p.sym], [grad_f])
        self.A_fun = ca.Function("A", [mpcc.w.sym, mpcc.p.sym], [jac_g])
        self.b_fun = ca.Function("b", [mpcc.w.sym, mpcc.p.sym], [mpcc.g.sym])
        self.G_fun = ca.Function("G", [mpcc.w.sym, mpcc.p.sym], [jac_G])
        self.g_fun = ca.Function("g", [mpcc.w.sym, mpcc.p.sym], [mpcc.G.sym])
        self.H_fun = ca.Function("H", [mpcc.w.sym, mpcc.p.sym], [jac_H])
        self.h_fun = ca.Function("h", [mpcc.w.sym, mpcc.p.sym], [mpcc.H.sym])
        
        self.Q = ca.DM(self.Q_sparsity)
        self.q = ca.DM.zeros(dims.nx)
        self.A = ca.DM(self.A_sparsity)
        self.b = ca.DM.zeros(dims.ng)
        self.G = ca.DM(self.G_sparsity)
        self.g = ca.DM.zeros(dims.ncc)
        self.H = ca.DM(self.H_sparsity)
        self.h = ca.DM.zeros(dims.ncc)
        self.lbx = np.copy(mpcc.w.lb)
        self.ubx = np.copy(mpcc.w.ub)
        self.lbg = np.copy(mpcc.g.lb)
        self.ubg = np.copy(mpcc.g.ub)

        self.solver = None


    def linearize(self, x0, lam_g=None, lam_G=None, lam_H=None, tr=np.inf):
        """
        Linearize the parent MPCC at the point given by `x0`, and optionally `lam_g`, `lam_G`, `lam_H`.
        Optionally also apply a ell infinity trust region constraint on the primal variables with radius `tr`.

        Todo:
           Implement convexification options!
        """
        self.mpcc.w.init = x0
        if lam_g is not None:
            self.mpcc.g.init_mult = lam_g
        else:
            self.mpcc.g.init_mult[:] = 0.0

        if lam_G is not None and self.use_mpcc_multipliers:
            self.mpcc.G.init_mult = lam_G
        else:
            self.mpcc.G.init_mult[:] = 0.0

        if lam_H is not None and self.use_mpcc_multipliers:
            self.mpcc.H.init_mult = lam_H
        else:
            self.mpcc.H.init_mult[:] = 0.0


        self.Q = self.Q_fun(self.mpcc.w.init, self.mpcc.g.init_mult, self.mpcc.G.init_mult, self.mpcc.H.init_mult, self.mpcc.p.val)
        self.q = self.q_fun(self.mpcc.w.init, self.mpcc.p.val)

        self.A = self.A_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.b = self.b_fun(self.mpcc.w.init, self.mpcc.p.val)

        self.G = self.G_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.g = self.g_fun(self.mpcc.w.init, self.mpcc.p.val)

        self.H = self.H_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.h = self.h_fun(self.mpcc.w.init, self.mpcc.p.val)

        # setup tr
        self.lbx = np.maximum(self.mpcc.w.lb - x0, -tr)
        self.ubx = np.minimum(self.mpcc.w.ub - x0, tr)


    def solve(self):
        """
        Solve the QPCC at the current linearization point.
        """
        pval = np.concat([
            self.Q.nonzeros(),
            self.q.full().flatten(),
            self.A.nonzeros(),
            self.b.full().flatten(),
            self.G.nonzeros(),
            self.g.full().flatten(),
            self.H.nonzeros(),
            self.h.full().flatten(),
        ])
        res = self.solver(p=pval, lbx=self.lbx, ubx=self.ubx)
        self.mpcc.w.res = res["w"]
        self.mpcc.w.mult = res["lam_x"]
        self.mpcc.g.val = res["g"]
        self.mpcc.g.mult = res["lam_g"]
        self.mpcc.G.val = res["G"]
        self.mpcc.H.val = res["H"]

    def create_solver(self, opts):
        """
        Create the solver for this QPCC.
        Currently only supports solving the QPCC via `mpccsol`.
        """
        if isinstance(opts, RegHomotopyOptions):
            self._build_mpccsol_solver("reg_homotopy", opts)
        else:
           raise NotImplementedError("Only the reg_homotopy plugin for mpccsol is currently supported.")
        

    def _build_mpccsol_solver(self, plugin, opts):
        """
        Build the mpccsol based solver for the QPCC.
        Currently this only supports only the `reg_homotopy_solver
        """
        Q = ca.SX.sym("Q", self.Q_sparsity)
        q = ca.SX.sym("q", self.q.size())
        A = ca.SX.sym("A", self.A_sparsity)
        b = ca.SX.sym("b", self.b.size())
        G = ca.SX.sym("G", self.G_sparsity)
        g = ca.SX.sym("g", self.g.size())
        H = ca.SX.sym("H", self.H_sparsity)
        h = ca.SX.sym("h", self.g.size())

        x = ca.SX.sym("x", self.dims.nx)

        f_expr = 0.5*ca.bilin(Q,x) + ca.dot(q,x)
        g_expr = A@x + b
        G_expr = G@x + g
        H_expr = H@x + h
        qpcc = {
            "x": x,
            "p": ca.vertcat(*(Q.nonzeros()),q, *(A.nonzeros()),b, *(G.nonzeros()),g, *(H.nonzeros()),h),
            "f": f_expr,
            "g": g_expr,
            "G": G_expr,
            "H": H_expr,
        }
        self.solver = mpccsol(plugin, qpcc, opts)
