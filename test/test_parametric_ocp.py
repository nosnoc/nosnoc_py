from examples.parametric_cart_pole_with_friction import solve_paramteric_example
from examples.cart_pole_with_friction import solve_example
import numpy as np

def main():
    ref_results = solve_example()
    results_parametric = solve_paramteric_example(with_global_var=False)

    assert np.allclose(ref_results["w_sol"], results_parametric["w_sol"], atol=1e-7)
    assert np.alltrue(ref_results["nlp_iter"] == results_parametric["nlp_iter"])
    assert results_parametric["v_global"].shape == (1, 0)
    assert ref_results["v_global"].shape == (1, 0)

    results_with_global_var = solve_paramteric_example(with_global_var=True)
    assert np.allclose(np.ones((1,)), results_with_global_var["v_global"], atol=1e-7)
    assert np.allclose(np.array(ref_results["x_list"]), np.array(results_with_global_var["x_list"]), atol=1e-7)



if __name__ == "__main__":
    main()
