import matplotlib.pyplot as plt
import nosnoc
import numpy as np

def plot_results(results):
    nosnoc.latexify_plot()

    x_traj = np.array(results["x_traj"])

    plt.figure()
    # states
    plt.subplot(3, 1, 1)
    plt.plot(results["t_grid"], x_traj[:, 0], label='$q_1$ - cart')
    plt.plot(results["t_grid"], x_traj[:, 1], label='$q_2$ - pole')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(results["t_grid"], x_traj[:, 2], label='$v_1$ - cart')
    plt.plot(results["t_grid"], x_traj[:, 3], label='$v_2$ - pole')
    plt.legend()
    plt.grid()

    # controls
    plt.subplot(3, 1, 3)
    plt.step(results["t_grid_u"], [results["u_traj"][0]] + results["u_traj"], label='u')

    plt.legend()
    plt.grid()

    plt.show()
