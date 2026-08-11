import unittest
from parameterized import parameterized
import nosnoc as ns
import numpy as np

from examples.generic_mpcc.generic_mpcc import create_generic_mpcc1, create_generic_mpcc2

reg_homotopy_parameters=[
    (problem,solution, homotopy_update_rule, homotopy_steering_strategy, objective_scaling_direct)
    for (problem,solution) in ((create_generic_mpcc1,np.array([1.0, 0.0])), (create_generic_mpcc2,np.array([1.43125000e+01, 5.62500000e-01, 1.43125000e+01, 5.62500000e-01, 1.87150604e-09, 1.79102370e-09, 1.90625001e+00, 1.98437500e+00])))
    for homotopy_update_rule in ns.mpccsol.plugins.reg_homotopy.HomotopyUpdateRule
    for homotopy_steering_strategy in ns.mpccsol.plugins.reg_homotopy.HomotopySteeringStrategy
    for objective_scaling_direct in (True, False)
    ]

class TestRegHomotopy(unittest.TestCase):

    @parameterized.expand(reg_homotopy_parameters)
    def test_basic_functionality(self,problem, solution, homotopy_update_rule, homotopy_steering_strategy, objective_scaling_direct):
        """
        Test basic reg_homotopy functionality.
        """
        solver_opts = ns.mpccsol.plugins.reg_homotopy.RegHomotopyOptions(
            assume_lower_bounds=False,
            homotopy_update_rule = homotopy_update_rule,
            homotopy_steering_strategy = homotopy_steering_strategy,
            objective_scaling_direct = objective_scaling_direct,
        )
        mpcc, init = problem()

        print(f"""
        Test args:
        homotopy_update_rule: {homotopy_update_rule}
        homotopy_steering_strategy: {homotopy_steering_strategy}
        objective_scaling_direct: {objective_scaling_direct}
        """)
        solver = ns.mpccsol.mpccsol("reg_homotopy", mpcc, solver_opts)
        sol = solver(**init)
        self.assertTrue(np.allclose(sol["w"], solution))
        self.assertTrue(np.all(np.minimum(sol["G"], sol["H"])<1e-6))
