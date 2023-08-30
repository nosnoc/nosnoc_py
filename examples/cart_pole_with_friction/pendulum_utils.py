import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nosnoc
import numpy as np

nosnoc.latexify_plot()
def plot_results(results):

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

    dt = results["t_grid"][1] - results["t_grid"][0]
    ani = animate_cart_pole(x_traj, dt=dt, saveas='cart_pole.gif')

    plt.show()


def animate_cart_pole(Xtraj, dt=0.03, saveas=None):
    '''
    Create animation of the cart pole system.
    Xtraj is a matrix of size N x 4, where N is the number of time steps, with the following columns:
    q1, q2, v1, v2

    dt defines the time gap (in seconds) between two succesive images.
    '''

    N = Xtraj.shape[0]
    pendulum_length = 1.0
    cart_width = .2
    cart_height =.1

    q1 = Xtraj[:, 0]
    q2 = Xtraj[:, 1]

    # x and y position of the tip of the pendulum
    pendu_tip_x = q1 + pendulum_length * np.sin(q2)
    pendu_tip_y = 0 - pendulum_length * np.cos(q2)

    xmin = min(np.min(q1), np.min(pendu_tip_x)) - 5 * cart_width / 2
    xmax = max(np.max(q1), np.max(pendu_tip_x)) + 5 * cart_width / 2

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()

        # level of cart
        ax.plot([xmin, xmax], [0, 0], 'k--')

        # draw rectancle for cart
        cart = mpl.patches.Rectangle((q1[i] - cart_width / 2, 0 - cart_height / 2), cart_width, cart_height, facecolor='C0')
        ax.add_patch(cart)

        # draw line for pendulum
        pendu = mpl.lines.Line2D([q1[i], pendu_tip_x[i]], [0, pendu_tip_y[i]], color='k', linewidth=2)
        ax.add_line(pendu)

        # trace of pendulum tip
        ax.plot(pendu_tip_x[:i], pendu_tip_y[:i], color='lightgray', linewidth=1)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([-1.2, 1.2])
        ax.set_aspect('equal')

    ani = FuncAnimation(fig, animate, N, interval=dt*1000, repeat_delay=500, repeat=True)
    if saveas is not None:
        ani.save(saveas, dpi=100)
    return ani

def plot_sigma_experiment(sigma_values, objective_values):

    plt.plot(sigma_values, objective_values)
    plt.grid()
    plt.xlabel(r'smoothing parameter $\sigma$')
    plt.ylabel(r'objective function value')
    plt.xscale('log')
    plt.show()
