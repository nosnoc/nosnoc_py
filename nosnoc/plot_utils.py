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


def plot_timings(timings, latexify=True, title=''):
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
        plt.bar(x, y_iter, bottom=y, label=f'homotopy iter {i}')
        y += y_iter
    plt.ylabel('CPU time [s]')
    plt.xlabel('simulation step')
    plt.legend()
    plt.xlim([-.5, Nsim - .5])
    plt.grid()
    plt.title(title)
    plt.show()
