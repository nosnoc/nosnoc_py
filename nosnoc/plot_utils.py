import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from nosnoc import NosnocProblem, get_results_from_primal_vector
from .utils import flatten_layer
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


def plot_timings(timings: np.ndarray, latexify=True, title=None, figure_filename=None):
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
    if title is not None:
        plt.title(title)
    if figure_filename is not None:
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    plt.show()


def plot_iterates(problem: NosnocProblem,
                  iterates: list,
                  latexify=False,
                  title_list=[],
                  figure_filename=None):

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
            plt.plot([x[i] for x in lambdas], label=f'$\\lambda_{i+1}$')
        for i in range(n_mu):
            plt.plot([x[i] for x in mus], label=f'$\\mu_{i+1}$')
        plt.grid(alpha=0.3)
        plt.title(title_list[it])
        plt.legend()

        # plot theta
        # TODO: make this step plot?
        plt.subplot(n_row, n_iterates, n_iterates * 1 + it + 1)
        for i in range(n_lam):
            plt.plot([x[i] for x in thetas], label=r'$\\theta_' + f'{i+1}$')
        plt.grid(alpha=0.3)
        plt.legend()

        # plot x
        x_list = results['x_all_list']
        plt.subplot(n_row, n_iterates, n_iterates * 2 + it + 1)
        for i in range(problem.model.dims.n_x):
            plt.plot([x[i] for x in x_list], label=r'$x_' + f'{i+1}$')
        plt.grid(alpha=0.3)
        plt.legend()

    if figure_filename is not None:
        plt.savefig(figure_filename)
        print(f'stored figure as {figure_filename}')

    # plt.show()
    return


def plot_voronoi_2d(Z, show=True, annotate=False, ax=None):
    """Plot voronoi regions."""
    from scipy.spatial import Voronoi, voronoi_plot_2d
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)
    vor = Voronoi(Z)
    fig = voronoi_plot_2d(vor, ax=ax)
    if ax is None:
        ax = fig.axes[0]

    if annotate:
        for i in range(Z.shape[0]):
            ax.text(Z[i, 0], Z[i, 1], f"p{i+1}")

    if show:
        plt.show()

    return ax


def scatter_3d(Z, show=True, ax=None, annotate=False):
    """
    3D scatter points.

    :param Z: List of points
    :param show: Show the scatterplot after drawing
    :param ax: Optional axis to draw the points onto
    :param annotate: Annotate the points (with p_x)
    :return: ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.scatter(
        [z[0] for z in Z],
        [z[1] for z in Z],
        [z[2] for z in Z],
    )

    if annotate:
        for i, pi in enumerate(Z):
            ax.text(pi[0], pi[1], pi[2], f"p{i+1}", zdir=(1, 0, 0))

    if show:
        plt.show()

    return ax


def _plot_color_hack_3d(ax, x, y, z, t):
    """Color hack for 3d plot."""
    t_min = min(t)
    dt = max(t) - t_min
    for i in range(np.size(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.hot(
            (t[i] - t_min) / dt
        ))


def plot_colored_line_3d(x, y, z, t, ax=None, label="Trajectory", label_4d="Time"):
    """
    Plot colored line in 3D.

    :param x: x values
    :param y: y values
    :param z: z values
    :param t: time values (fourth dimension)
    :param ax: axis
    :param label: Label of the line
    :param label_4d: Label of the 4th dimension (None = don't add)
    :return ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    _plot_color_hack_3d(ax, x, y, z, t)
    im = ax.scatter(x, y, z, c=t,
                    label=label, cmap=plt.hot())

    if label_4d is not None:
        im.set_label(label_4d)
        plt.colorbar(im, ax=ax)

    return ax
