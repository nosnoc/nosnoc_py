from matplotlib import pyplot as plt
import numpy as np
from oscilator_example import get_oscilator_model, X_SOL, TSIM
import nosnoc
import pickle
import os
import itertools

BENCHMARK_DATA_PATH = 'private_oscilator_benchmark_data'
SCHEMES = [nosnoc.IrkSchemes.GAUSS_LEGENDRE, nosnoc.IrkSchemes.RADAU_IIA]

NS_VALUES = [1, 2, 3, 4]
NFE_VALUES = [2]
NSIM_VALUES = [1, 5, 9, 10, 20, 40, 60, 100]
USE_FESD_VALUES = [True, False]

TOL = 1e-12


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


def get_results_filename(opts: nosnoc.NosnocOpts):
    filename = 'oscilator_bm_results_'
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


def get_opts(Nsim, n_s, N_fe, scheme, use_fesd):
    opts = nosnoc.NosnocOpts()
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep
    opts.comp_tol = TOL
    opts.sigma_N = TOL
    opts.irk_scheme = scheme
    opts.print_level = 0
    opts.tol_ipopt = TOL

    opts.n_s = n_s
    opts.N_finite_elements = N_fe
    opts.pss_mode = nosnoc.PssMode.STEP
    opts.use_fesd = use_fesd
    return opts


def run_benchmark():
    for n_s, N_fe, Nsim, scheme, use_fesd in itertools.product(NS_VALUES, NFE_VALUES, NSIM_VALUES, SCHEMES, USE_FESD_VALUES):

        model = get_oscilator_model()
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



def count_failures(results):
    status_list: list = results['status']
    return len([x for x in status_list if x != nosnoc.Status.SUCCESS])



def order_plot():
    nosnoc.latexify_plot()
    N_fe = 2

    linestyles = ['-', '--', '-.', ':', ':', '-.', '--', '-']
    marker_types = ['o', 's', 'v', '^', '>', '<', 'd', 'p']
    SCHEME = nosnoc.IrkSchemes.GAUSS_LEGENDRE
    # SCHEME = nosnoc.IrkSchemes.RADAU_IIA

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
                error = np.max(np.abs(x_end - X_SOL))
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
    plt.legend()
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=1.0)

    fig_filename = f'oscilator_benchmark_{SCHEME.name}.pdf'
    plt.savefig(fig_filename, bbox_inches='tight')
    print(f"Saved figure to {fig_filename}")
    plt.show()


if __name__ == "__main__":
    # generate data
    # run_benchmark()

    # evalute
    order_plot()
