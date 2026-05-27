import unittest
import casadi as ca
import numpy as np
import nosnoc


class TestAutoModel(unittest.TestCase):
    """
    Test auto model class
    """

    def test_find_nonlinear_components(self):
        x = ca.SX.sym('x')

        f_nonsmooth_ode = x + ca.sin(x) + x*ca.cos(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        results = am._find_nonlinear_components(am.f_nonsmooth_ode)
        self.assertEqual(len(results), 3)

        f_nonsmooth_ode = -(x + ca.sin(x) + x*ca.cos(x))
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        results = am._find_nonlinear_components(am.f_nonsmooth_ode)
        self.assertEqual(len(results), 3)

        f_nonsmooth_ode = x - ca.sin(x) + x*(-ca.cos(x))
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        results = am._find_nonlinear_components(am.f_nonsmooth_ode)
        self.assertEqual(len(results), 3)

    def test_check_additive(self):
        x = ca.SX.sym('x')

        f_nonsmooth_ode = x*ca.sign(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        additive = am._check_additive(am.f_nonsmooth_ode)
        self.assertTrue(additive)

        f_nonsmooth_ode = ca.fmax(x, 5)*ca.sign(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        additive = am._check_additive(am.f_nonsmooth_ode)
        self.assertFalse(additive)

    def test_check_smooth(self):
        x = ca.SX.sym('x')

        f_nonsmooth_ode = x
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        smooth = am._check_smooth(am.f_nonsmooth_ode)
        self.assertTrue(smooth)

        f_nonsmooth_ode = x*ca.sign(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        smooth = am._check_smooth(am.f_nonsmooth_ode)
        self.assertFalse(smooth)

        f_nonsmooth_ode = ca.fmax(x, 5)*ca.sign(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        smooth = am._check_smooth(am.f_nonsmooth_ode)
        self.assertFalse(smooth)

    def test_rebuild_nonlin(self):
        x = ca.SX.sym('x')

        f_nonsmooth_ode = x
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        f = am._rebuild_nonlin(am.f_nonsmooth_ode)
        self.assertEqual(f.str(), f_nonsmooth_ode.str())

        f_nonsmooth_ode = ca.sign(x)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        f = am._rebuild_nonlin(am.f_nonsmooth_ode)
        self.assertEqual(len(am.alpha), 1)
        self.assertEqual(len(am.c), 1)
        self.assertEqual(am.c[0].str(), 'x')
        self.assertEqual(f.str(), '((2*alpha_0)-1)')

        f_nonsmooth_ode = ca.fmax(x, 5)
        am = nosnoc.NosnocAutoModel(x=x, f_nonsmooth_ode=f_nonsmooth_ode, x0=np.array([]), u=ca.SX([]))
        f = am._rebuild_nonlin(am.f_nonsmooth_ode)
        self.assertEqual(len(am.alpha), 1)
        self.assertEqual(len(am.c), 1)
        self.assertEqual(am.c[0].str(), '(x-5)')
        self.assertEqual(f.str(), '((alpha_0*x)+(5*(1-alpha_0)))')
