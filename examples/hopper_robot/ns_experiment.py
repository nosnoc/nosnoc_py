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
N_STAGES = 50
TERMINAL_TIME = 5.0
N_EXPR = [2, 3]


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
        results_sparse = [e[0] for e in sparse.get()]
        results_dense = [e[0] for e in dense.get()]
        
    # pickle
    with open(strftime("%Y-%m-%d-%H-%M-%S-ns-experiment.pkl", gmtime()), 'wb') as f:
        experiment_results = {'results_sparse': results_sparse,
                              'results_dense': results_dense,
                              'cpu_times_sparse': cpu_times_sparse,
                              'cpu_times_dense': cpu_times_dense,
                              'n_expr': n_expr}
        pickle.dump(experiment_results, f)
    breakpoint()
    plot_for_paper(cpu_times_sparse, cpu_times_dense, n_expr)


def ns_experiment():
    # Try running solver with multiple n_s with both dense and sparse S
    cpu_times_dense = []
    cpu_times_sparse = []
    results_dense = []
    results_sparse = []
    
    n_expr = N_EXPR
    for n_s in n_expr:
        opts = get_default_options_step()
        opts.terminal_time = TERMINAL_TIME
        opts.N_stages = N_STAGES
        opts.n_s = n_s
        results = solve_ocp_step(opts=opts, plot=True, x_goal=X_GOAL)
        cpu_times_sparse.append(sum(results['cpu_time_nlp']))
        results_sparse.append(results)
        

    for n_s in n_expr:
        opts = get_default_options()
        opts.terminal_time = TERMINAL_TIME
        opts.N_stages = N_STAGES
        opts.n_s = n_s
        opts.pss_mode = ns.PssMode.STEWART
        results = solve_ocp(opts=opts, plot=False, dense=True, ref_as_init=False, x_goal=X_GOAL)
        cpu_times_dense.append(sum(results['cpu_time_nlp']))
        results_dense.append(results)

    # pickle
    with open(strftime("%Y-%m-%d-%H-%M-%S-ns-experiment.pkl", gmtime()), 'wb') as f:
        experiment_results = {'results_sparse': results_sparse,
                              'results_dense': results_dense,
                              'cpu_times_sparse': cpu_times_sparse,
                              'cpu_times_dense': cpu_times_dense,
                              'n_expr': n_expr}
        pickle.dump(experiment_results, f)
    breakpoint()
    plot_for_paper(cpu_times_sparse, cpu_times_dense, n_expr)


def plot_for_paper(cpu_times_sparse, cpu_times_dense, n_expr):
    ns.latexify_plot()
    plt.figure()
    plt.plot(n_expr, cpu_times_sparse,'Xb-', label="Step")
    plt.plot(n_expr, cpu_times_dense, 'Xr-', label="Stewart")
    plt.xlabel('$n_s$')
    plt.ylabel('cpu time [s]')
    plt.legend(loc='best')
    
    plt.show()
    
if __name__ == '__main__':
    #ns_experiment()
    ns_experiment_mp()
