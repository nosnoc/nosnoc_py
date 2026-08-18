from examples.cart_pole_with_friction.main_cart_pole_mpc import (
    main,
)
from parameterized import parameterized
import unittest
import nosnoc as ns
import numpy as np

# (rti, prepare_step, n_advanced_steps)
options = [
    (False, ns.rtopt.PreparationStep.NONE, 0),
    (True, ns.rtopt.PreparationStep.NONE, 0),
    (True, ns.rtopt.PreparationStep.FULL, 0),
    (True, ns.rtopt.PreparationStep.SQPCC, 3),
]

class MPCTest(unittest.TestCase):

    @parameterized.expand(options)
    def test_cartpole_mpc(self, rti, prepare_step, n_advanced_steps):
        x_res,u_res,t_grid,control_grid = main(rti=rti, prepare_step=prepare_step, n_advanced_steps=n_advanced_steps)
        self.assertTrue(abs(x_res[-1,1] - np.pi) <= 0.2) # loose tolerance
