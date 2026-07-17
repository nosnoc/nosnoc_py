from examples.Acary2014.two_gene import (
    get_default_options,
    TSIM,
    X0,
    solve_two_gene,
    get_two_gene_model,
)
from parameterized import parameterized
import unittest
import nosnoc
import numpy as np

options = [
    (lifting,rk_representation, rk_scheme, dcs_mode)
    for lifting in [True,False]
    for dcs_mode in nosnoc.DcsMode
    for rk_scheme in nosnoc.RKScheme
    for rk_representation in nosnoc.RKRepresentation
]

X_SOL = np.array([8,8])

def compute_errors(integrator) -> dict:
    X_sim = integrator.get("x")
    t_grid = integrator.get_time_grid()

    err_x_end = np.max(np.abs(X_sim[-1,:] - X_SOL))
    return {
        "x_end": err_x_end,
    }


class TwoGeneTests(unittest.TestCase):

    @parameterized.expand(options)
    def test_two_gene(self, lifting, rk_representation, rk_scheme, dcs_mode):
        opts = get_default_options(
            rk_representation=rk_representation,
            rk_scheme=rk_scheme,
            dcs_mode=dcs_mode,
        )
        opts.print_level = 0
        model = get_two_gene_model(X0, lifting)
        _,_,_,_,integrator = solve_two_gene(model=model, opts=opts)
        errors = compute_errors(integrator)

        print(errors)
        tol = 1e-5
        assert errors["x_end"] < tol


if __name__ == "__main__":
    unittest.main()
