import nosnoc as ns
import casadi as ca
import numpy as np
import pickle
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from time import gmtime, strftime
from hopper_ocp_step import solve_ocp_step, get_default_options_step
from hopper_ocp import solve_ocp, get_default_options

X_GOAL = 3.0
TERMINAL_TIME = 5.0
N_S = 2
N_EXPR = [50, 60, 70, 80, 90, 100]


def run_sparse(n_stages):
    opts = get_default_options_step()
    opts.terminal_time = TERMINAL_TIME
    opts.N_stages = n_stages
    opts.n_s = N_S
    results = solve_ocp_step(opts=opts, plot=False, x_goal=X_GOAL)
    return results, sum(results['cpu_time_nlp']), sum(results['nlp_iter'])

def run_dense(n_stages):
    opts = get_default_options()
    opts.terminal_time = TERMINAL_TIME
    opts.N_stages = n_stages
    opts.n_s = N_S
    opts.pss_mode = ns.PssMode.STEWART
    results = solve_ocp(opts=opts, plot=False, dense=True, ref_as_init=False, x_goal=X_GOAL)
    return results, sum(results['cpu_time_nlp']), sum(results['nlp_iter'])
    
def stage_experiment_mp():
    # Try running solver with multiple Nfe with both dense and sparse S
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
    with open(strftime("%Y-%m-%d-%H-%M-%S-n-stages-experiment.pkl", gmtime()), 'wb') as f:
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

def plot_for_paper(cpu_times_sparse, cpu_times_dense, nlp_iter_sparse, nlp_iter_dense, n_expr):
    ns.latexify_plot()
    plt.figure()
    plt.plot(n_expr, cpu_times_sparse,'Xb-', label="Step")
    plt.plot(n_expr, cpu_times_dense, 'Xr-', label="Stewart")
    plt.xlabel('$n_{\mathrm{stages}}$')
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
    stage_experiment_mp()
