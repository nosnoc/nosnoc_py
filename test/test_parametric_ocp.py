from examples.parametric_cart_pole_with_friction import solve_paramteric_example
from examples.cart_pole_with_friction import solve_example
import numpy as np

def main():
    ref_results = solve_example()
    results_parametric = solve_paramteric_example()

    assert np.allclose(ref_results["w_sol"], results_parametric["w_sol"], atol=1e-7)
    assert np.alltrue(ref_results["nlp_iter"] == results_parametric["nlp_iter"])


if __name__ == "__main__":
    main()
