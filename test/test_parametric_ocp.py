from examples.parametric_cart_pole_with_friction import solve_example
import numpy as np

def main():
    ref_results = solve_example(parameter_variant=0)
    results_p_mat = solve_example(parameter_variant=1)

    assert np.allclose(ref_results["w_sol"], results_p_mat["w_sol"], atol=1e-7)
    assert np.alltrue(ref_results["nlp_iter"] == results_p_mat["nlp_iter"])


if __name__ == "__main__":
    main()
