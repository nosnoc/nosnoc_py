# Some tools to plot the pickled results in order to make sure our experiments are converging to good results

import nosnoc as ns
import casadi as ca
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from functools import partial

def init_func(htrail, ftrail):
    htrail.set_data([], [])
    ftrail.set_data([], [])
    return htrail, ftrail

def animate_robot(state, head, foot, body, ftrail, htrail):
    x_head, y_head = state[0], state[1]
    x_foot, y_foot = state[0] - state[3]*np.sin(state[2]), state[1] - state[3]*np.cos(state[2])
    head.set_offsets([x_head, y_head])
    foot.set_offsets([x_foot, y_foot])
    body.set_data([x_foot, x_head], [y_foot, y_head])

    ftrail.set_data(np.append(ftrail.get_xdata(orig=False), x_foot), np.append(ftrail.get_ydata(orig=False), y_foot))
    htrail.set_data(np.append(htrail.get_xdata(orig=False), x_head), np.append(htrail.get_ydata(orig=False), y_head))
    return head, foot, body, ftrail, htrail


def plot_results(results):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.1, 1.1)
    head = ax.scatter([0], [0], color='b', s=[100])
    foot = ax.scatter([0], [0], color='r', s=[50])
    body, = ax.plot([], [], 'k')
    ftrail, = ax.plot([], [], color='r', alpha=0.5)
    htrail, = ax.plot([], [], color='b', alpha=0.5)
    ani = FuncAnimation(fig, partial(animate_robot, head=head, foot=foot, body=body, htrail=htrail, ftrail=ftrail),
                        init_func=partial(init_func, htrail=htrail, ftrail=ftrail),
                        frames=results['x_traj'], blit=True)
    # try:
    #     ani.save('hopper.gif', writer='imagemagick', fps=10)
    # except Exception:
    #     print("install imagemagick to save as gif")

    # Plot Trajectory
    plt.figure()
    x_traj = np.array(results['x_traj'])
    t = x_traj[:, -1]
    x = x_traj[:, 0]
    y = x_traj[:, 1]
    theta = x_traj[:, 2]
    leg_len = x_traj[:, 3]
    plt.subplot(4, 1, 1)
    plt.plot(results['t_grid'], t)
    plt.subplot(4, 1, 2)
    plt.plot(results['t_grid'], x)
    plt.plot(results['t_grid'], y)
    plt.subplot(4, 1, 3)
    plt.plot(results['t_grid'], theta)
    plt.subplot(4, 1, 4)
    plt.plot(results['t_grid'], leg_len)
    # Plot Controls
    plt.figure()
    u_traj = np.array(results['u_traj'])
    reaction = u_traj[:, 0]
    leg_force = u_traj[:, 1]
    slack = u_traj[:, 2]
    sot = u_traj[:, 3]
    plt.subplot(4, 1, 1)
    plt.step(results['t_grid_u'], np.concatenate((reaction, [reaction[-1]])))
    plt.subplot(4, 1, 2)
    plt.step(results['t_grid_u'], np.concatenate((leg_force, [leg_force[-1]])))
    plt.subplot(4, 1, 3)
    plt.step(results['t_grid_u'], np.concatenate((slack, [slack[-1]])))
    plt.subplot(4, 1, 4)
    plt.step(results['t_grid_u'], np.concatenate((sot, [sot[-1]])))
    plt.show()

def plot_pickle(fname):
    with open(fname, 'rb') as f:
        experiment_results = pickle.load(f)
    sparse_results = experiment_results['results_sparse']
    dense_results = experiment_results['results_dense']

    for result in sparse_results:
        plot_results(result)
    for result in dense_results:
        plot_results(result)

if __name__ == '__main__':
    plot_pickle(sys.argv[1])
