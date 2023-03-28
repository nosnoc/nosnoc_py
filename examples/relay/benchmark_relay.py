from matplotlib import pyplot as plt
import numpy as np
from relay_feedback_system import get_relay_feedback_system_model, get_default_options, TSIM
import nosnoc
import pickle
import os
import itertools

BENCHMARK_DATA_PATH = 'private_relay_benchmark_data'
REF_RESULTS_FILENAME = 'relay_benchmark_results.pickle'

# SCHEMES = [nosnoc.IrkSchemes.GAUSS_LEGENDRE, nosnoc.IrkSchemes.RADAU_IIA]
SCHEMES = [nosnoc.IrkSchemes.RADAU_IIA]
NS_VALUES = [2]
NFE_VALUES = [2]
NSIM_VALUES = [120]

SOLVER_CLASSES = [
                nosnoc.NaiveIpoptSolver,
                nosnoc.NosnocSolver,
                nosnoc.NosnocMIpoptSolver,
                nosnoc.NosnocCustomSolver,
                ]

TOL = 1e-6

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

def generate_reference_solution():
    opts = get_default_options()
    opts.n_s = 5
    opts.N_finite_elements = 3

    Nsim = 500
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep
    opts.comp_tol = TOL * 1e-2
    opts.sigma_N = TOL * 1e-2
    opts.do_polishing_step = False
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN

    model = get_relay_feedback_system_model()

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, print_level=1)
    looper.run()
    results = looper.get_results()
    results['opts'] = opts

    pickle_results(results, REF_RESULTS_FILENAME)


def get_results_filename(SolverClass: nosnoc.NosnocSolverBase, opts: nosnoc.NosnocOpts):
    filename = 'relay_bm_'
    filename += SolverClass.__name__ + '_'
    filename += 'Nfe_' + str(opts.N_finite_elements) + '_'
    filename += 'ns' + str(opts.n_s) + '_'
    filename += 'tol' + str(opts.comp_tol) + '_'
    filename += 'dt' + str(opts.terminal_time) + '_'
    filename += 'Tsim' + str(TSIM) + '_'
    filename += '.pickle'
    return filename


def get_opts(Nsim, n_s, N_fe, scheme):
    opts = get_default_options()
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep
    opts.comp_tol = TOL
    opts.sigma_N = TOL
    opts.irk_scheme = scheme
    opts.print_level = 0

    opts.n_s = n_s
    opts.N_finite_elements = N_fe
    # opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
    # opts.irk_representation = nosnoc.IrkRepresentation.DIFFERENTIAL
    return opts


def run_benchmark():
    for SolverClass, n_s, N_fe, Nsim, scheme in itertools.product(SOLVER_CLASSES, NS_VALUES, NFE_VALUES, NSIM_VALUES, SCHEMES):

        model = get_relay_feedback_system_model()
        opts = get_opts(Nsim, n_s, N_fe, scheme)
        solver = SolverClass(opts, model)

        # loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, print_level=1)
        looper.run()
        results = looper.get_results()
        results['opts'] = opts
        filename = get_results_filename(SolverClass, opts)
        pickle_results(results, filename)
        del solver, looper, results, model, opts


def get_reference_solution():
    return unpickle_results(REF_RESULTS_FILENAME)

def count_failures(results):
    status_list: list = results['status']
    return len([x for x in status_list if x != nosnoc.Status.SUCCESS])

def evaluate_reference_solution():
    results = get_reference_solution()
    n_fail = count_failures(results)

    print(f"Reference solution got {n_fail} failing subproblems")


def compare_plot():
    nosnoc.latexify_plot()

    n_s = NS_VALUES[0]
    scheme = SCHEMES[0]
    N_fe = NFE_VALUES[0]
    Nsim = NSIM_VALUES[0]

    n_cmp = len(SOLVER_CLASSES)

    fig, ax = plt.subplots()

    metric_per_step = n_cmp * [None]
    metric = 'cpu_nlp'
    # metric = 'nlp_iter'

    # load data
    x_ref = get_reference_solution()['X_sim'][-1]
    for i, SolverClass in enumerate(SOLVER_CLASSES):
        opts = get_opts(Nsim, n_s, N_fe, scheme)
        filename = get_results_filename(SolverClass, opts)
        results = unpickle_results(filename)
        failures = count_failures(results)
        x_end = results['X_sim'][-1]
        error = np.max(np.abs(x_end - x_ref))
        print(f"{SolverClass.__name__} failed {failures} times, error wrt exact solution: {error:.2e}")

        metric_per_step[i] = np.sum(results[metric], 1)


    # matplotlib boxplot
    x = np.arange(n_cmp)
    ax.boxplot(metric_per_step, positions=x, showfliers=True, widths=0.6)
    ax.grid(axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels([SolverClass.__name__ for SolverClass in SOLVER_CLASSES], rotation=15)

    if metric == 'nlp_iter':
        ax.set_ylabel('NLP iterations')
    elif metric == 'cpu_nlp':
        ax.set_ylabel('CPU time [s]')
    ax.legend()
    fig_filename = f'relay_bm_{metric}_ns_{n_s}.pdf'
    plt.savefig(fig_filename, bbox_inches='tight')
    print(f"Saved figure to {fig_filename}")
    plt.show()


if __name__ == "__main__":
    # generate data
    # generate_reference_solution()
    # run_benchmark()

    # evalute
    # evaluate_reference_solution()
    compare_plot()
