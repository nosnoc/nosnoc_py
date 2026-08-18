import unittest
from parameterized import parameterized
import numpy as np
import nosnoc as ns
from examples.sliding_mode_ocp.sliding_mode_ocp import (
    get_default_options,
    get_sliding_mode_ocp_description
)

rk_options = [
    (rk_representation, rk_scheme, use_fesd)
    for rk_representation in ns.RKRepresentation
    for rk_scheme in ns.RKScheme
    for use_fesd in (True,False)
]


class TestShiftWarmstart(unittest.TestCase):

    @parameterized.expand(rk_options)
    def test_stewart(self, rk_representation, rk_scheme, use_fesd):
        """
        Test Stewart DTP shift warmstart routine.
        """
        N_stages = 5
        N_fe = 2
        n_s = 2
        opts = get_default_options(
            N_stages = N_stages,
            N_finite_elements = 3,
            n_s = 2,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme,
            dcs_mode = ns.DcsMode.STEWART,
            use_fesd=use_fesd,
        )
        solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        model = get_sliding_mode_ocp_description()

        solver = ns.OcpSolver(model, opts, solver_opts)

        np.copyto(solver.dtp.w.res,np.random.rand(len(solver.dtp.w))) # Don't need to solve just add random values :P
        solver.warmstart_shift()
        # Check depth 3 variables
        for name in ["x", "z", "lam", "theta", "mu"]:
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii,:,:].init
                res = var[ii+1,:,:].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages,:,:].init
            res = var[N_stages,:,:].res
            self.assertTrue(np.allclose(init,res))
        # check depth 2 variables
        for name in (["h"] if opts.use_fesd else []):
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii,:].init
                res = var[ii+1,:].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages,:].init
            res = var[N_stages,:].res
            self.assertTrue(np.allclose(init,res))

        # check depth 1 variables
        for name in ["u"]:
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii].init
                res = var[ii+1].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages].init
            res = var[N_stages].res
            self.assertTrue(np.allclose(init,res))


    @parameterized.expand(rk_options)
    def test_heaviside(self, rk_representation, rk_scheme, use_fesd):
        """
        Test Heaviside DTP shift warmstart routine.
        """
        N_stages = 5
        N_fe = 2
        n_s = 2
        opts = get_default_options(
            N_stages = N_stages,
            N_finite_elements = 3,
            n_s = 2,
            rk_representation = rk_representation,
            rk_scheme = rk_scheme,
            dcs_mode = ns.DcsMode.STEP,
            use_fesd=use_fesd,
        )
        solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        model = get_sliding_mode_ocp_description()

        solver = ns.OcpSolver(model, opts, solver_opts)

        np.copyto(solver.dtp.w.res,np.random.rand(len(solver.dtp.w))) # Don't need to solve just add random values :P
        solver.warmstart_shift()
        # Check depth 3 variables
        for name in ["x", "z", "lambda_n", "lambda_p", "alpha"]:
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii,:,:].init
                res = var[ii+1,:,:].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages,:,:].init
            res = var[N_stages,:,:].res
            self.assertTrue(np.allclose(init,res))
        # check depth 2 variables
        for name in (["h"] if opts.use_fesd else []):
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii,:].init
                res = var[ii+1,:].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages,:].init
            res = var[N_stages,:].res
            self.assertTrue(np.allclose(init,res))

        # check depth 1 variables
        for name in ["u"]:
            var = getattr(solver.dtp.w,name)
            for ii in range(1,N_stages):
                init = var[ii].init
                res = var[ii+1].res
                self.assertTrue(np.allclose(init,res))
            init = var[N_stages].init
            res = var[N_stages].res
            self.assertTrue(np.allclose(init,res))
