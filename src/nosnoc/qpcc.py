from copy import copy
from dataclasses import dataclass

import casadi as ca

@dataclass
class QpccDims():
    nw: int
    ng: int
    ncc: int
    
class Qpcc():
    """
    Build a parametric QPCC with the given sparsity pattern.
    """
    def __init__(self, mpcc, use_mpcc_multipliers=False):
        self.mpcc = copy(mpcc)
        self.use_mpcc_multipliers = mpcc
        dims = QpccDims(nw=len(mpcc.w), ng=len(mpcc.g), ncc=len(mpcc.G))
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
        self.q_fun = ca.Function("q", [mpcc.w.sym, lam_g, lam_G, lam_H, lam_g,mpcc. p.sym], [nabla_L])
        self.A_fun = ca.Function("A", [mpcc.w.sym, mpcc.p.sym], [jac_g])
        self.b_fun = ca.Function("b", [mpcc.w.sym, mpcc.p.sym], [mpcc.g.sym])
        self.G_fun = ca.Function("G", [mpcc.w.sym, mpcc.p.sym], [jac_G])
        self.g_fun = ca.Function("g", [mpcc.w.sym, mpcc.p.sym], [mpcc.G.sym])
        self.H_fun = ca.Function("H", [mpcc.w.sym, mpcc.p.sym], [jac_H])
        self.h_fun = ca.Function("h", [mpcc.w.sym, mpcc.p.sym], [mpcc.H.sym])
        
        self.Q = ca.DM(self.Q_sparsity)
        self.q = ca.DM.zeros(dims.nw)
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
        self.lbG = np.copy(mpcc.G.lb)
        self.lbH = np.copy(mpcc.H.lb)
        

    def linearize(self, x0=None, lam_g=None, lam_G=None, lam_H=None, tr=1.0):
        if x0:
            self.mpcc.w.init = x0
        if lam_g:
            self.mpcc.g.init_mult = lam_g
        if lam_G and self.use_mpcc_multipliers:
            self.mpcc.G.init_mult = lam_G
        if lam_H and self.use_mpcc_multipliers:
            self.mpcc.H.init_mult = lam_H
        

        self.Q = self.Q_fun(self.mpcc.w.init, self.mpcc.g.init_mult, self.mpcc.G.init_mult, self.mpcc.H.init_mult, self.mpcc.p.val)
        self.q = self.q_fun(self.mpcc.w.init, self.mpcc.g.init_mult, self.mpcc.G.init_mult, self.mpcc.H.init_mult, self.mpcc.p.val)

        self.A = self.A_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.b = self.b_fun(self.mpcc.w.init, self.mpcc.p.val)

        self.G = self.G_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.g = self.g_fun(self.mpcc.w.init, self.mpcc.p.val)

        self.H = self.H_fun(self.mpcc.w.init, self.mpcc.p.val)
        self.h = self.h_fun(self.mpcc.w.init, self.mpcc.p.val)
