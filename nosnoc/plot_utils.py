import numpy as np

import matplotlib.pyplot as plt
import matplotlib


def latexify_plot():
    params = {
        # "backend": "TkAgg",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def plot_timings(timings, latexify=True, title='', figure_filename=''):
    # latexify plot
    if latexify:
        latexify_plot()

    # timings = [sum(t) for t in timings]
    nlp_iter = timings.shape[1]
    Nsim = timings.shape[0]
    x = range(Nsim)
    y = np.zeros((Nsim,))
    plt.figure()
    for i in range(nlp_iter):
        y_iter = timings[:, i]
        plt.bar(x, y_iter, bottom=y, label=f'homotopy iter {i+1}')
        y += y_iter
    x_range = [-.5, Nsim - .5]
    mean_cpu = np.mean(np.sum(timings, axis=1))
    plt.plot(x_range, mean_cpu*np.ones(2,), label=f'mean {mean_cpu:.3f}', linestyle=':', color='black')
    plt.ylabel('CPU time [s]')
    plt.xlabel('simulation step')
    plt.legend()
    plt.xlim(x_range)
    plt.grid(alpha=0.3)
    plt.title(title)
    if figure_filename != '':
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    plt.show()
