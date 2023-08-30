from cart_pole_with_friction import solve_example
from parametric_cart_pole_with_friction import solve_paramteric_example
from pendulum_utils import plot_results

def main():
    # results = solve_example()
    results = solve_paramteric_example(with_global_var=True)
    plot_results(results)


if __name__ == "__main__":
    main()
