import nosnoc as ns
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import gmtime, strftime
from cart_pole_with_friction import solve_example
import sys

N_EXPR = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def run_step(n_stages):
    results = solve_example(N_stages=n_stages, pss_mode=ns.PssMode.STEP)
    return results, sum(filter(None, results['cpu_time_nlp'])), sum(filter(None, results['nlp_iter']))


def run_stewart(n_stages):
    results = solve_example(N_stages=n_stages, pss_mode=ns.PssMode.STEWART)
    return results, sum(filter(None, results['cpu_time_nlp'])), sum(filter(None, results['nlp_iter']))


def stage_experiment_mp():
    # Try running solver with multiple Nfe with both stewart and step S
    cpu_times_stewart = []
    cpu_times_step = []
    nlp_iter_stewart = []
    nlp_iter_step = []
    results_stewart = []
    results_step = []

    n_expr = N_EXPR
    with Pool(cpu_count() - 2) as p:
        step = p.map_async(run_step, n_expr)
        stewart = p.map_async(run_stewart, n_expr)
        step.wait()
        stewart.wait()
        cpu_times_step = [e[1] for e in step.get()]
        cpu_times_stewart = [e[1] for e in stewart.get()]
        nlp_iter_step = [e[2] for e in step.get()]
        nlp_iter_stewart = [e[2] for e in stewart.get()]
        results_step = [e[0] for e in step.get()]
        results_stewart = [e[0] for e in stewart.get()]

    # pickle
    with open(strftime("%Y-%m-%d-%H-%M-%S-n-stages-experiment-cart-pole.pkl", gmtime()), 'wb') as f:
        experiment_results = {'results_step': results_step,
                              'results_stewart': results_stewart,
                              'cpu_times_step': cpu_times_step,
                              'cpu_times_stewart': cpu_times_stewart,
                              'nlp_iter_step': nlp_iter_step,
                              'nlp_iter_stewart': nlp_iter_stewart,
                              'n_expr': n_expr}
        pickle.dump(experiment_results, f)
    plot_for_paper(cpu_times_step, cpu_times_stewart, nlp_iter_step, nlp_iter_stewart, n_expr)


def plot_from_pickle(fname):
    with open(fname, 'rb') as f:
        experiment_results = pickle.load(f)
        plot_for_paper(experiment_results['cpu_times_step'],
                       experiment_results['cpu_times_stewart'],
                       experiment_results['nlp_iter_step'],
                       experiment_results['nlp_iter_stewart'],
                       experiment_results['n_expr'])


def plot_for_paper(cpu_times_step, cpu_times_stewart, nlp_iter_step, nlp_iter_stewart, n_expr):
    ns.latexify_plot()
    plt.figure()
    plt.plot(n_expr, np.array(cpu_times_step), 'Xb-', label="Step")
    plt.plot(n_expr, np.array(cpu_times_stewart), 'Xr-', label="Stewart")
    plt.xlabel('$N$')
    plt.ylabel('cpu time [s]')
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))

    plt.figure()
    plt.plot(n_expr, np.array(cpu_times_step)/np.array(nlp_iter_step), 'Xb-', label="Step")
    plt.plot(n_expr, np.array(cpu_times_stewart)/np.array(nlp_iter_stewart), 'Xr-', label="Stewart")
    plt.xlabel('$N$')
    plt.ylabel(r'cpu time/iteration [$s$]')
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_from_pickle(sys.argv[1])
    else:
        stage_experiment_mp()
