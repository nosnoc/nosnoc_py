import numpy as np
import casadi as ca

from .nosnoc_types import IrkScheme


def rk4(f, x0, tf: float, n_steps: int = 1):
    # Compute time step from final time and number of steps
    dt = tf / n_steps

    # Create time vector
    t = np.linspace(0, tf, n_steps + 1)

    # Create storage for solution
    x = np.zeros((n_steps + 1, len(x0)))
    x[0, :] = x0

    # Runge-Kutta 4th order method
    for i in range(n_steps):
        k1 = f(x[i, :])
        k2 = f(x[i, :] + dt * k1 / 2)
        k3 = f(x[i, :] + dt * k2 / 2)
        k4 = f(x[i, :] + dt * k3)
        x[i + 1, :] = x[i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4).full().flatten()

    x = x.tolist()
    return x, t


def rk4_on_timegrid(f, x0, t_grid: np.ndarray) -> list:

    x = np.zeros((len(t_grid) + 1, len(x0)))
    x[0, :] = x0

    # Runge-Kutta 4th order method
    for i in range(len(t_grid)):
        # TODO: multiple steps?
        dt = t_grid[i]
        k1 = f(x[i, :])
        k2 = f(x[i, :] + dt * k1 / 2)
        k3 = f(x[i, :] + dt * k2 / 2)
        k4 = f(x[i, :] + dt * k3)
        x[i + 1, :] = x[i, :] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4).full().flatten()

    x = x.tolist()
    return x


if __name__ == "__main__":
    # test RK tableaus
    for irk_scheme in IrkSchemes:
        n_s = 2
        B, C, D, tau_root = generate_butcher_tableu_integral(n_s, irk_scheme)
        print(f"Tableau for {n_s=} {irk_scheme.name} reads")
        print(f"{B=}\n{C=}\n{D=}\n{tau_root=}\n\n")
