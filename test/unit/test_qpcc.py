import unittest
from parameterized import parameterized
import nosnoc as ns
import vdx
from vdx.vartypes import *
import numpy as np


class TestQPCC(unittest.TestCase):

    def _create_basic_mpcc(self):
        mpcc = ns.MPCC()
        mpcc.w.x[()] = Primal("x", 2)
        mpcc.w.y[()] = Primal("y", 1, lb=-10, ub=10)
        mpcc.g.g[()] = Constraint(mpcc.w.x[()].sym[0]**2 - mpcc.w.y[()].sym**2 + mpcc.w.x[()].sym[0]*mpcc.w.y[()].sym, ub = 1.0, lb=-np.inf)
        mpcc.G.CC[()] = CConstraint(mpcc.w.x[()].sym[0])
        mpcc.H.CC[()] = CConstraint(mpcc.w.x[()].sym[1])
        mpcc.f = ca.bilin(np.array([[1,0.0],[0.0,1]]), mpcc.w.x[()].sym[0:2] - np.array([1.0, 2.0])) - 0.1*mpcc.w.y[()].sym
        return mpcc

    def test_independence(self):
        """
        Test qpcc vectors are independent from the oritinal mpcc.
        """
        mpcc = self._create_basic_mpcc()
        qpcc = ns.Qpcc(mpcc)
        # Check we don't accidentally have a dependence on w
        for vecname in ("lb", "ub", "init", "res", "mult"):
            self.assertTrue(getattr(mpcc.w,vecname) is not getattr(qpcc.mpcc.w, vecname))
        # Check we don't accidentally have a dependence on g
        for vecname in ("lb", "ub", "init_mult", "val", "mult"):
            self.assertTrue(getattr(mpcc.g,vecname) is not getattr(qpcc.mpcc.g, vecname))


    def test_sparsities(self):
        """
        Test qpcc sparsity is set correctly.
        """
        mpcc = self._create_basic_mpcc()
        qpcc = ns.Qpcc(mpcc)
        self.assertTrue(qpcc.Q_sparsity.nnz() == 5)
        self.assertTrue(qpcc.A_sparsity.nnz() == 2)
        self.assertTrue(qpcc.G_sparsity.nnz() == 1)
        self.assertTrue(qpcc.H_sparsity.nnz() == 1)

    def test_linearization(self):
        """
        Test qpcc linearization functionality.
        """
        mpcc = self._create_basic_mpcc()
        mpcc.g.g[()](mult=1.0)
        qpcc = ns.Qpcc(mpcc)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult)
        self.assertAlmostEqual(qpcc.Q[2,0],1.0)
        self.assertAlmostEqual(qpcc.Q[0,0],4.0)
        self.assertAlmostEqual(qpcc.Q[1,1],2.0)
        self.assertAlmostEqual(qpcc.Q[2,2],-2.0)
        qpcc.linearize(mpcc.w.res, lam_g=None)
        self.assertAlmostEqual(qpcc.Q[2,0],0.0)
        self.assertAlmostEqual(qpcc.Q[0,0],2.0)
        self.assertAlmostEqual(qpcc.Q[1,1],2.0)
        self.assertAlmostEqual(qpcc.Q[2,2],0.0)

    def test_convexification(self):
        mpcc = self._create_basic_mpcc()
        mpcc.g.g[()](mult=1.0)
        qpcc = ns.Qpcc(mpcc)
        # Default options with no convexification
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult, cvx_opts=ns.ConvexificationOptions())
        self.assertAlmostEqual(qpcc.Q[2,0],1.0)
        self.assertAlmostEqual(qpcc.Q[0,0],4.0)
        self.assertAlmostEqual(qpcc.Q[1,1],2.0)
        self.assertAlmostEqual(qpcc.Q[2,2],-2.0)
        cvx_opts=ns.ConvexificationOptions(mode=ns.ConvexificationMode.LEVENBERG_MARQUARDT ,lambda_lm=10.0)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult, cvx_opts=cvx_opts)
        self.assertAlmostEqual(qpcc.Q[2,0],1.0)
        self.assertAlmostEqual(qpcc.Q[0,0],14.0)
        self.assertAlmostEqual(qpcc.Q[1,1],12.0)
        self.assertAlmostEqual(qpcc.Q[2,2],8.0)
        self.assertTrue(qpcc.Q_sparsity.is_equal(qpcc.Q_sparsity_orig))

        mpcc.g.g[()](mult=10.0)
        cvx_opts = ns.ConvexificationOptions(mode=ns.ConvexificationMode.PROJECT)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult, cvx_opts=cvx_opts)
        self.assertAlmostEqual(qpcc.Q[2,0],5.2149, places=3)
        self.assertAlmostEqual(qpcc.Q[0,0],23.0811, places=3)
        self.assertAlmostEqual(qpcc.Q[1,1],2.0, places=3)
        self.assertAlmostEqual(qpcc.Q[2,2],1.17827, places=3)
        eigvals, eigvectors = np.linalg.eig(qpcc.Q.full())
        self.assertTrue(np.all(eigvals >= cvx_opts.eps_hessian-0.1*cvx_opts.eps_hessian)) # Check eigenvalues with some tolerance

        mpcc.g.g[()](mult=10.0)
        cvx_opts = ns.ConvexificationOptions(mode=ns.ConvexificationMode.MIRROR)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult, cvx_opts=cvx_opts)
        self.assertAlmostEqual(qpcc.Q[2,0],0.429934, places=3)
        self.assertAlmostEqual(qpcc.Q[0,0],24.1623, places=3)
        self.assertAlmostEqual(qpcc.Q[1,1],2.0, places=3)
        self.assertAlmostEqual(qpcc.Q[2,2],22.3565, places=3)
        eigvals, eigvectors = np.linalg.eig(qpcc.Q.full())
        self.assertTrue(np.all(eigvals >= cvx_opts.eps_hessian-0.1*cvx_opts.eps_hessian)) # Check eigenvalues with some tolerance

        # TODO(@anton) also test GERSHGORIN, but practically this is not a useful approach.

    def test_solve(self):
        mpcc = self._create_basic_mpcc()
        qpcc = ns.Qpcc(mpcc)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult)
        qpcc.update_bounds()
        qpcc.create_solver(ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions())
        qpcc.solve()
        dx = qpcc.get_dx()
        y = qpcc.get_y()
        self.assertAlmostEqual(dx[0],0.0, places=6)
        self.assertAlmostEqual(dx[1],2.0, places=6)
        self.assertAlmostEqual(dx[2],10.0, places=6)
        self.assertAlmostEqual(y[0],0.0, places=6)
        mpcc.w.res += dx
        np.copyto(mpcc.w.mult,qpcc.mpcc.w.mult)
        np.copyto(mpcc.g.mult,qpcc.mpcc.g.mult)
        qpcc.linearize(mpcc.w.res, lam_g=mpcc.g.mult)
        qpcc.update_bounds()
        qpcc.solve()
        dx = qpcc.get_dx()
        y = qpcc.get_y()
        self.assertAlmostEqual(dx[0],0.0, places=6)
        self.assertAlmostEqual(dx[1],0.0, places=6)
        self.assertAlmostEqual(dx[2],-5.0, places=6)
        self.assertAlmostEqual(y[0],-0.005, places=6)
