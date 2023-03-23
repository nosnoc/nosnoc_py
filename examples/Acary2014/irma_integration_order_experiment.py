from matplotlib import pyplot as plt
import numpy as np
from irma import get_irma_model, get_default_options, SWITCH_ON, LIFTING, X0
import nosnoc
import pickle
import os
import itertools

BENCHMARK_DATA_PATH = 'private_irma_benchmark_data'
REF_RESULTS_FILENAME = 'irma_benchmark_results.pickle'
SCHEMES = [nosnoc.IrkSchemes.GAUSS_LEGENDRE, nosnoc.IrkSchemes.RADAU_IIA]

NS_VALUES = [1, 2, 3, 4]
NFE_VALUES = [3]
# NSIM_VALUES = [1, 3, 10, 20, 50, 100, 300] # convergence issues for Legendre
NSIM_VALUES = [1, 3, 9, 18, 50, 100, 300]
USE_FESD_VALUES = [True, False]
# USE_FESD_VALUES = [False]


# # NOTE: this has convergence issues
# NS_VALUES = [2]
# NSIM_VALUES = [10]
# SCHEME = nosnoc.IrkSchemes.GAUSS_LEGENDRE


TOL = 1e-12
TSIM = 100


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

    model = get_irma_model(SWITCH_ON, LIFTING)

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, print_level=1)
    looper.run()
    results = looper.get_results()
    results['opts'] = opts

    pickle_results(results, REF_RESULTS_FILENAME)


def get_results_filename(opts):
    filename = 'irma_bm_results_'
    filename += 'Nfe_' + str(opts.N_finite_elements) + '_'
    filename += 'ns' + str(opts.n_s) + '_'
    filename += 'tol' + str(opts.comp_tol) + '_'
    filename += 'dt' + str(opts.terminal_time) + '_'
    filename += 'Tsim' + str(TSIM) + '_'
    filename += opts.irk_scheme.name
    if not opts.use_fesd:
        filename += '_nofesd'
    filename += '.pickle'
    return filename


def get_opts(Nsim, n_s, N_fe, scheme, use_fesd):
    opts = get_default_options()
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep
    opts.comp_tol = TOL
    opts.sigma_N = TOL
    opts.irk_scheme = scheme
    opts.print_level = 1
    opts.use_fesd = use_fesd

    opts.n_s = n_s
    opts.N_finite_elements = N_fe
    # opts.step_equilibration = nosnoc.StepEquilibrationMode.DIRECT
    # opts.irk_representation = nosnoc.IrkRepresentation.DIFFERENTIAL
    return opts


def run_benchmark():
    for n_s, N_fe, Nsim, scheme, use_fesd in itertools.product(NS_VALUES, NFE_VALUES, NSIM_VALUES, SCHEMES, USE_FESD_VALUES):

        model = get_irma_model(SWITCH_ON, LIFTING)
        opts = get_opts(Nsim, n_s, N_fe, scheme, use_fesd)
        solver = nosnoc.NosnocSolver(opts, model)

        # loop
        looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim, print_level=1)
        looper.run()
        results = looper.get_results()
        results['opts'] = opts
        filename = get_results_filename(opts)
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


def order_plot():
    nosnoc.latexify_plot()
    N_fe = 3
    ref_results = get_reference_solution()
    x_end_ref = ref_results['X_sim'][-1]

    linestyles = ['-', '--', '-.', ':', ':', '-.', '--', '-']
    marker_types = ['o', 's', 'v', '^', '>', '<', 'd', 'p']
    # SCHEME = nosnoc.IrkSchemes.RADAU_IIA
    SCHEME = nosnoc.IrkSchemes.RADAU_IIA

    ax = plt.figure()
    for use_fesd in [True, False]:
        for i, n_s in enumerate(NS_VALUES):
            errors = []
            step_sizes = []
            for Nsim in NSIM_VALUES:
                opts = get_opts(Nsim, n_s, N_fe, SCHEME, use_fesd)
                filename = get_results_filename(opts)
                results = unpickle_results(filename)
                print(f"loading filde {filename}")
                x_end = results['X_sim'][-1]
                n_fail = count_failures(results)
                error = np.max(np.abs(x_end - x_end_ref))
                print("opts.n_s: ", opts.n_s, "opts.terminal_time: ", opts.terminal_time, "error: ", error, "n_fail: ", n_fail)
                errors.append(error)
                step_sizes.append(opts.terminal_time)

            label = r'$n_s=' + str(n_s) +'$'
            if results['opts'].irk_scheme == nosnoc.IrkSchemes.RADAU_IIA:
                if n_s == 1:
                    label = 'implicit Euler: 1'
                else:
                    label = 'Radau IIA: ' + str(2*n_s-1)
            elif results['opts'].irk_scheme == nosnoc.IrkSchemes.GAUSS_LEGENDRE:
                label = 'Gauss-Legendre: ' + str(2*n_s)
            if use_fesd:
                label += ', FESD'
            else:
                label += ', Standard'
            plt.plot(step_sizes, errors, label=label, marker=marker_types[i], linestyle=linestyles[i])
    plt.grid()
    plt.xlabel('Step size')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.xscale('log')
    # plt.legend(loc='center left')
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, .97), ncol=2, framealpha=1.0)

    fig_filename = f'irma_benchmark_{SCHEME.name}.pdf'
    plt.savefig(fig_filename, bbox_inches='tight')
    print(f"Saved figure to {fig_filename}")
    plt.show()


if __name__ == "__main__":
    # generate data
    run_benchmark()
    generate_reference_solution()

    # evalute
    evaluate_reference_solution()
    order_plot()
