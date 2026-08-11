import unittest
from parameterized import parameterized
import numpy as np
import nosnoc
from examples.sliding_mode_ocp.sliding_mode_ocp import (
    get_default_options,
    get_sliding_mode_ocp_description
)

rk_options = [
    (rk_representation, rk_scheme)
    for rk_representation in nosnoc.RKRepresentation
    for rk_scheme in nosnoc.RKScheme
]

class TestOcpSolver(unittest.TestCase):

    @parameterized.expand(rk_options)
    def test_get(self, rk_representation, rk_scheme):
        """
        Test whether OcpSolver.get() returns the correct size in all rk scheme cases.
        """
        N_stages = 5
        N_fe = 3
        n_s = 2
        opts = get_default_options(
            N_stages = N_stages,
            N_finite_elements = 3,
            n_s = 2,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme
        )
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        model = get_sliding_mode_ocp_description()

        solver = nosnoc.OcpSolver(model, opts, solver_opts)

        x = solver.get("x")
        u = solver.get("u")
        t_grid = solver.get_time_grid()
        control_grid = solver.get_control_grid()

        with self.subTest("length(x) = time grid dimensions"):
            self.assertEqual(x.shape[0], t_grid.shape[0])
        with self.subTest("length(u)+1 = control grid dimensions"):
            self.assertEqual(u.shape[0]+1, control_grid.shape[0])

    @parameterized.expand(rk_options)
    def test_get_time_grid_full(self, rk_representation, rk_scheme):
        """
        Test whether OcpSolver.get_time_grid_full() returns the correct size in all rk scheme cases.
        """
        N_stages = 5
        N_fe = 3
        n_s = 2
        opts = get_default_options(
            N_stages = N_stages,
            N_finite_elements = N_fe,
            n_s = n_s,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme
        )
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        model = get_sliding_mode_ocp_description()

        solver = nosnoc.OcpSolver(model, opts, solver_opts)

        t_grid = solver.get_time_grid_full()

        self.assertEqual((N_stages*N_fe*(n_s + solver.dtp.rbp)) + 1, t_grid.shape[0])

    def test_set(self):
        """
        Test whether OcpSolver.set() returns the correct size in all rk scheme cases.
        """
        N_stages = 5
        N_fe = 3
        n_s = 2
        opts = get_default_options(
            N_stages = N_stages,
            N_finite_elements = 3,
            n_s = 2
        )
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        model = get_sliding_mode_ocp_description()

        solver = nosnoc.OcpSolver(model, opts, solver_opts)

        solver.set("x", (0,0,n_s), lb=-10, ub=10)
        solver.set("u", (1,), lb=-20, ub=20)
        solver.set("u", (range(2,N_stages),), lb=3)
        solver.set("u", (slice(4,None),), lb=5)

        self.assertTrue(np.allclose(solver.dtp.w.u[2:4].lb, 3.0))
        self.assertTrue(np.allclose(solver.dtp.w.u[4:].lb, 5.0))
        self.assertTrue(np.allclose(solver.dtp.w.x[0,0,n_s].lb, -10))
        self.assertTrue(np.allclose(solver.dtp.w.x[0,0,n_s].ub, 10))
        self.assertTrue(np.allclose(solver.dtp.w.u[1].lb, -20))
        self.assertTrue(np.allclose(solver.dtp.w.u[1].ub, 20))
