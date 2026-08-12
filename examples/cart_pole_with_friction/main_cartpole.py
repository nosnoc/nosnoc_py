from parametric_cart_pole_with_friction import solve_paramteric_example
from pendulum_utils import plot_results

def main():
    solver = solve_paramteric_example(with_global_var=False)
    plot_results(solver)


if __name__ == "__main__":
    main()
