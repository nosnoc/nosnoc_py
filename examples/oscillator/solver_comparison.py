from matplotlib import pyplot as plt
import numpy as np
from oscillator_example import get_oscillator_model, get_default_options, X_SOL, TSIM
import nosnoc
import pickle
import os
import itertools


BENCHMARK_DATA_PATH = 'private_oscillator_benchmark_data'

SOLVER_CLASSES = [
                nosnoc.NaiveIpoptSolver,
                nosnoc.NosnocSolver,
                nosnoc.NosnocMIpoptSolver,
                nosnoc.NosnocCustomSolver,
                ]
# SOLVER_CLASSES = [nosnoc.NosnocSolver, nosnoc.NosnocCustomSolver]
# SOLVER_CLASSES = [nosnoc.NosnocSolver]
NS_VALUES = [2]
TOL = 1e-8
Nsim = 29

def pickle_results(results, filename):
    # create directory if it does not exist
    if not os.path.exists(BENCHMARK_DATA_PATH):
        os.makedirs(BENCHMARK_DATA_PATH)
    # save
    file = os.path.join(BENCHMARK_DATA_PATH, filename)
    with open(file, 'wb') as f:
        pickle.dump(results, f)

def unpickle_results(filename):
    file = os.path.join(BENCHMARK_DATA_PATH, filename)
    with open(file, 'rb') as f:
        results = pickle.load(f)
    return results


def get_results_filename(SolverClass: nosnoc.NosnocSolverBase, opts: nosnoc.NosnocOpts):
    filename = 'osciallator_'
    filename += SolverClass.__name__ + '_'
    filename += 'Nfe_' + str(opts.N_finite_elements) + '_'
    filename += 'ns' + str(opts.n_s) + '_'
    filename += 'tol' + str(opts.comp_tol) + '_'
    filename += 'dt' + str(opts.terminal_time) + '_'
    filename += 'Tsim' + str(TSIM) + '_'
    filename += opts.irk_scheme.name + '_'
    filename += opts.pss_mode.name
    if not opts.use_fesd:
        filename += '_nofesd'
    filename += '.pickle'
    return filename


def get_opts(n_s):
    opts = get_default_options()

    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep
    opts.comp_tol = TOL
    opts.sigma_N = TOL
    opts.print_level = 0
    opts.tol_ipopt = TOL
    opts.pss_mode = nosnoc.PssMode.STEP
    opts.n_s = n_s

    return opts


def run_benchmark():
    for SolverClass, n_s in itertools.product(SOLVER_CLASSES, NS_VALUES):

        model = get_oscillator_model()
        opts = get_opts(n_s)
        solver: nosnoc.NosnocSolverBase = SolverClass(opts, model)

        # loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, print_level=1)
        looper.run()
        results = looper.get_results()
        results['opts'] = opts
        filename = get_results_filename(SolverClass, opts)
        pickle_results(results, filename)
        del solver, looper, results, model, opts


def count_failures(results):
    status_list: list = results['status']
    return len([x for x in status_list if x != nosnoc.Status.SUCCESS])


def compare_plot():
    nosnoc.latexify_plot()

    plot_sim_index = 18
    n_s = NS_VALUES[0]

    n_cmp = len(SOLVER_CLASSES)

    fig, ax = plt.subplots()

    metric_list = n_cmp * [None]

    # metric = 'nlp_iter'
    metric = 'cpu_nlp'

    # load data
    for i, SolverClass in enumerate(SOLVER_CLASSES):
        opts = get_opts(n_s)
        filename = get_results_filename(SolverClass, opts)
        results = unpickle_results(filename)
        failures = count_failures(results)
        x_end = results['X_sim'][-1]
        error = np.max(np.abs(x_end - X_SOL))
        print(f"{SolverClass.__name__} failed {failures} times, error wrt exact solution: {error:.2e}")

        metric_list[i] = results[metric][plot_sim_index]

    hom_iter = metric_list[-1].shape[0]

    print(f"metric_list = {metric_list}")
    alpha_sim = results['alpha_sim'][plot_sim_index]
    if np.max(np.abs(alpha_sim[0] - alpha_sim[-1])) > .1:
        print("got problem with switch")


    # plot
    x = range(n_cmp)
    metric_values = np.array(metric_list)
    y = np.zeros((n_cmp,))
    for i in range(hom_iter):
        y_iter = metric_values[:, i]
        ax.bar(x, y_iter, bottom=y, label=f'homotopy iter {i+1}')
        y += y_iter
    ax.grid(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels([SolverClass.__name__ for SolverClass in SOLVER_CLASSES], rotation=15)

    if metric == 'nlp_iter':
        ax.set_ylabel('NLP iterations')
    elif metric == 'cpu_nlp':
        ax.set_ylabel('CPU time [s]')
    ax.legend()
    fig_filename = f'oscillator_solver_comparison_{metric}_ns_{n_s}.pdf'
    plt.savefig(fig_filename, bbox_inches='tight')
    print(f"Saved figure to {fig_filename}")
    plt.show()


if __name__ == "__main__":
    # generate data
    run_benchmark()

    # evalute
    compare_plot()
