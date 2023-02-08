import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from nosnoc import NosnocProblem, get_results_from_primal_vector
from .utils import flatten_layer, flatten
from .nosnoc_types import PssMode


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


def plot_timings(timings: np.ndarray, latexify=True, title='', figure_filename=''):
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
    plt.plot(x_range,
             mean_cpu * np.ones(2,),
             label=f'mean/step {mean_cpu:.3f}',
             linestyle=':',
             color='black')
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


def plot_iterates(problem: NosnocProblem,
                  iterates: list,
                  latexify=False,
                  title_list=[],
                  figure_filename=''):

    # latexify plot
    if latexify:
        latexify_plot()

    plt.figure()
    n_iterates = len(iterates)

    if title_list == []:
        title_list = n_iterates * ['']

    if problem.opts.pss_mode != PssMode.STEWART:
        raise NotImplementedError

    n_row = 3
    for it, iterate in enumerate(iterates):
        results = get_results_from_primal_vector(problem, iterate)

        # flatten sys layer
        lambdas = flatten_layer(results['lambda_list'], 2)
        thetas = flatten_layer(results['theta_list'], 2)
        mus = flatten_layer(results['mu_list'], 2)

        # flatten fe layer
        lambdas = flatten_layer(lambdas, 0)
        thetas = flatten_layer(thetas, 0)
        mus = flatten_layer(mus, 0)

        n_lam = len(lambdas[0])
        n_mu = len(mus[0])

        # plot lambda, mu
        plt.subplot(n_row, n_iterates, n_iterates * 0 + it + 1)
        for i in range(n_lam):
            plt.plot([x[i] for x in lambdas], label=f'$\lambda_{i+1}$')
        for i in range(n_mu):
            plt.plot([x[i] for x in mus], label=f'$\mu_{i+1}$')
        plt.grid(alpha=0.3)
        plt.title(title_list[it])
        plt.legend()

        # plot theta
        # TODO: make this step plot?
        plt.subplot(n_row, n_iterates, n_iterates * 1 + it + 1)
        for i in range(n_lam):
            plt.plot([x[i] for x in thetas], label=r'$\theta_' + f'{i+1}$')
        plt.grid(alpha=0.3)
        plt.legend()

        # plot x
        x_list = results['x_all_list']
        plt.subplot(n_row, n_iterates, n_iterates * 2 + it + 1)
        for i in range(problem.model.dims.n_x):
            plt.plot([x[i] for x in x_list], label=r'$x_' + f'{i+1}$')
        plt.grid(alpha=0.3)
        plt.legend()

    if figure_filename != '':
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    # plt.show()
    return


def plot_matrix_and_qr(matrix):
    import matplotlib.pyplot as plt
    from nosnoc.plot_utils import latexify_plot
    latexify_plot()
    fig, axs = plt.subplots(1, 3)

    axs[0].spy(matrix)
    axs[1].set_title('A')

    Q, R = np.linalg.qr(matrix)
    axs[1].spy(Q)
    axs[1].set_title('Q')
    axs[2].spy(R)
    axs[2].set_title('R')
    plt.show()


def spy_magnitude_plot(matrix, ax=None, fig=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None):
    rows, cols = matrix.nonzero()
    values = np.abs(matrix[rows, cols])

    cmap = matplotlib.colormaps['inferno']
    # scatter spy
    sc = ax.scatter(cols, rows, c=values, cmap=cmap,
                        norm=matplotlib.colors.LogNorm(vmin=np.min(values), vmax=np.max(values)),
                        marker='s', s=20)
    ax.set_xlim((-0.5, matrix.shape[1] - 0.5))
    ax.set_ylim((matrix.shape[0] - 0.5, -0.5))
    ax.set_aspect('equal')
    fig.colorbar(sc, ax = ax, ticks=matplotlib.ticker.LogLocator())

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
