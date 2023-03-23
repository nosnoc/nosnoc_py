import nosnoc as ns
import casadi as ca
import numpy as np
import pickle
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import gmtime, strftime
from hopper_ocp_step import solve_ocp_step, get_default_options_step
from hopper_ocp import solve_ocp, get_default_options

X_GOAL = 3.0
N_STAGES = 50
TERMINAL_TIME = 5.0
N_EXPR = [2, 3, 4, 5, 6, 7]


def run_sparse(n_s):
    opts = get_default_options_step()
    opts.terminal_time = TERMINAL_TIME
    opts.N_stages = N_STAGES
    opts.n_s = n_s
    results = solve_ocp_step(opts=opts, plot=False, x_goal=X_GOAL)
    return results, sum(results['cpu_time_nlp'])

def run_dense(n_s):
    opts = get_default_options()
    opts.terminal_time = TERMINAL_TIME
    opts.N_stages = N_STAGES
    opts.n_s = n_s
    opts.pss_mode = ns.PssMode.STEWART
    results = solve_ocp(opts=opts, plot=False, dense=True, ref_as_init=False, x_goal=X_GOAL)
    return results, sum(results['cpu_time_nlp'])
    
def ns_experiment_mp():
    # Try running solver with multiple n_s with both dense and sparse S
    cpu_times_dense = []
    cpu_times_sparse = []
    nlp_iter_dense = []
    nlp_iter_sparse = []
    results_dense = []
    results_sparse = []
    
    n_expr = N_EXPR
    with Pool(cpu_count() - 1) as p:
        sparse = p.map_async(run_sparse, n_expr)
        dense = p.map_async(run_dense, n_expr)
        sparse.wait()
        dense.wait()
        cpu_times_sparse = [e[1] for e in sparse.get()]
        cpu_times_dense = [e[1] for e in dense.get()]
        nlp_iter_sparse = [e[2] for e in sparse.get()]
        nlp_iter_dense = [e[2] for e in dense.get()]
        results_sparse = [e[0] for e in sparse.get()]
        results_dense = [e[0] for e in dense.get()]
        
    # pickle
    with open(strftime("%Y-%m-%d-%H-%M-%S-ns-experiment.pkl", gmtime()), 'wb') as f:
        experiment_results = {'results_sparse': results_sparse,
                              'results_dense': results_dense,
                              'cpu_times_sparse': cpu_times_sparse,
                              'cpu_times_dense': cpu_times_dense,
                              'nlp_iter_sparse': nlp_iter_sparse,
                              'nlp_iter_dense': nlp_iter_dense,
                              'n_expr': n_expr}
        pickle.dump(experiment_results, f)
    breakpoint()
    plot_for_paper(cpu_times_sparse, cpu_times_dense, nlp_iter_sparse, nlp_iter_dense, n_expr)

def plot_from_pickle(fname):
    with open(fname, 'rb') as f:
        experiment_results = pickle.load(f)
        plot_for_paper(experiment_results['cpu_times_sparse'],
                       experiment_results['cpu_times_dense'],
                       experiment_results['nlp_iter_sparse'],
                       experiment_results['nlp_iter_dense'],
                       experiment_results['n_expr'])

def plot_for_paper(cpu_times_sparse, cpu_times_dense, nlp_iter_sparse, nlp_iter_dense, n_expr):
    ns.latexify_plot()
    plt.figure()
    plt.plot(n_expr, cpu_times_sparse,'Xb-', label="Step")
    plt.plot(n_expr, cpu_times_dense, 'Xr-', label="Stewart")
    plt.xlabel('$n_s$')
    plt.ylabel('cpu time [s]')
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.figure()
    plt.plot(n_expr, np.array(cpu_times_sparse)/np.array(nlp_iter_sparse),'Xb-', label="Step")
    plt.plot(n_expr, np.array(cpu_times_dense)/np.array(nlp_iter_dense), 'Xr-', label="Stewart")
    plt.xlabel('$n_{\mathrm{stages}}$')
    plt.ylabel('cpu time/iteration [s/iter]')
    plt.legend(loc='best')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_from_pickle(sys.argv[1])
    else:
        ns_experiment_mp()
