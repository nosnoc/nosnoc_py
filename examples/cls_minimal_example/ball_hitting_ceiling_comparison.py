"""
`ccopt` against `reg_homotopy`, over FESD-J taken apart piece by piece, for the ball-hitting-ceiling
problem.

The model and the analytic reference are taken from `ball_hitting_ceiling_1d`; the cases differ only
in the impact discretization and the MPCC solver. Every case is swept over `N_FE_SWEEP` finite
elements per step (one integrator step, `h` free down to `MIN_H_FRACTION*h_0`), and one table per
`N_fe` compares them on

  * the time spent in the solver, and
  * how far the velocity trajectory ends up from the analytic one.

Where the impact ends up is most of what the table shows. FESD-J carries impulse variables, so the
velocity jumps *at* a finite element boundary and the impact is exact. The relaxed OC has no
impulses; the impact is a large contact force integrated over one element, which is only plastic in
the limit, so the solver has to collapse that element and the residual post impact velocity is O(h).

What FESD-J adds on top of the relaxed OC is three separable pieces, and `DISCRETIZATIONS` below
switches them on one at a time so the table attributes the difference:

  1. `RELAXED_OC_IMPULSE`: the impulse `Lambda_normal` and Newton's restitution law at the
     boundaries. This only *offers* the exact impact; the smeared one stays feasible next to it. At
     a boundary the trajectory reaches with `v- = 0` (because the impact already happened inside the
     previous element) restitution degenerates to `0 = 0` and constrains nothing, so this piece
     alone is a statement about which solution the solver is drawn to, not about which ones exist.
  2. `FESD_J`: adds the pairs `lambda_normal[i,j,k] perp Y_gap[i,j]`. Radau IIA has no collocation
     point at an element's left boundary, so without them an element may carry a huge contact force
     while the gap at its *start* is positive -- exactly the smeared impact. This is the piece that
     makes it infeasible.
  3. `eps_cls`: the Eq. (18) non-penetration constraint, emitted by `FESD_J` only. The `FESD_J` row
     is run both with and without it, so the last two rows isolate what it contributes.

`RELAXED_OC_IMPULSE_ONLY` attacks the same smeared impact from the other side: instead of adding
(2) to forbid it, it deletes `lambda_normal` outright, so the dynamics are free flight between
boundaries and there is no force to smear. Its `lambda max` column is therefore always 0, and its
MPCC is the smallest of the four.

Run it in the `env_ccopt` environment, which is the only one carrying `libcasadi_nlpsol_ccopt`
next to a working `ipopt`/`ma27`, see the export block at the end of `env_ccopt/bin/activate`.

    python examples/cls_minimal_example/ball_hitting_ceiling_comparison.py [--no-plot]

Each case runs in its own subprocess. That is not cosmetic: once `ccopt` has been built in a
process, a subsequent IPOPT solve in the *same* process dies with `Restoration_Failed` after a
single iteration per homotopy step, which would make `reg_homotopy` look like it fails on a problem
it actually solves. The isolation makes the result independent of the order in `CASES`.
"""

import json
import os
import subprocess
import sys
import time
from statistics import median

import numpy as np
import matplotlib.pyplot as plt

# The subprocess re-execs this file by absolute path, so the sibling example has to be importable
# no matter what the working directory is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nosnoc
import ball_hitting_ceiling_1d as ceiling

X0 = ceiling.X0
N_SIM = ceiling.N_SIM

# Number of finite elements per control stage to sweep over. Every case is run at every value, and
# the stats are printed as one table per N_fe. Note that this moves the nominal element length
# h_0 = h_sim/N_fe, and with it the floor below.
N_FE_SWEEP = tuple(range(2, 11))

# The finite element length may shrink to MIN_H_FRACTION*h_0, which is what lets FESD place a
# (nearly) zero length element on the impact.
MIN_H_FRACTION = 1.0e-10

# (impact discretization, eps_cls, MPCC solver) per case. The names must match
# `nosnoc.ClsDiscretization` and the `mpccsol` plugins. The order is irrelevant thanks to the
# subprocess isolation, see the module docstring. Trim the tuples to run fewer cases.
#
# The four rows below take FESD-J's finite element boundary block apart one piece at a time; see the
# module docstring for what each piece does.
DISCRETIZATIONS = (
    #  cls_discretization      eps_cls
    ("RELAXED_OC",             0.0),    # no impulses at all; eps_cls is a no-op
    ("RELAXED_OC_IMPULSE",     0.0),    # + impulse and restitution
    ("RELAXED_OC_IMPULSE_ONLY",0.0),    # ... but with the contact force removed instead
    ("FESD_J",                 0.0),    # + lambda_normal perp Y_gap
    ("FESD_J",                 1e-3),   # + Eq. (18), i.e. full FESD-J
)
SOLVERS = ("reg_homotopy", "ccopt")
CASES = tuple(disc + (solver,) for solver in SOLVERS for disc in DISCRETIZATIONS)

# Timed repeats per case, run after a warm-up solve. The warm-up matters for `ccopt`, whose first
# call pays the MadNLP build cost (measured ~7.5 s end to end against ~3.2 s afterwards).
N_REPEATS = 3

# Marks the one line of the child's stdout that carries the result. IPOPT and MadNLP write to
# stdout at the C level, so the JSON cannot simply be the whole output.
RESULT_SENTINEL = "__NOSNOC_RESULT__"


def get_solver_options(solver):
    """
    MPCC solver options per plugin.

    Note that `homotopy_update_slope`, `N_homotopy` and `complementarity_tol` live on
    `RegHomotopyOptions` only; `CCOptOptions` carries `madnlp_opts` and `ccopt_opts` and nothing
    else, so setting them there would be silently ignored.
    """
    if solver == "ccopt":
        solver_opts = nosnoc.mpccsol.plugins.ccopt.CCOptOptions()
        solver_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        solver_opts.madnlp_opts["print_level"] = "Error"
        # The plain MadNLP defaults do not get through this problem at all: the relaxation stalls
        # around a constraint violation of 1e-1. The roll-off update together with the tight
        # tolerance and a `sigma_min` below the h floor is what drives the contact element down.
        solver_opts.madnlp_opts["tol"] = 1e-6
        #solver_opts.madnlp_opts["max_iter"] = 3000
        solver_opts.ccopt_opts["relaxation_update.TYPE"] = "RolloffRelaxationUpdate"
        #solver_opts.ccopt_opts["relaxation_update.sigma_min"] = 1e-12
    else:
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        solver_opts.opts_casadi_nlp["ipopt"]["linear_solver"] = "ma27"
        solver_opts.opts_casadi_nlp["ipopt"]["print_level"] = 0
        solver_opts.opts_casadi_nlp["print_time"] = 0
        solver_opts.print_level = 0
        solver_opts.opts_casadi_nlp["ipopt"]["max_iter"] = 30000
        solver_opts.homotopy_update_slope = 0.2
        solver_opts.N_homotopy = 15
        # The relaxed OC satisfies the complementarity only up to O(h) on the contact element, so
        # the homotopy cannot be driven as far as it can for FESD-J.
        solver_opts.complementarity_tol = 1e-6
    return solver_opts


def get_integrator_options(solver, **kwargs):
    default_args = {
        "T_sim": ceiling.T_SIM,
        "N_sim": N_SIM,
        "solver_opts": get_solver_options(solver),
        "print_level": 0,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def _solver_iterations(solver, stats):
    """A single number of solver iterations, whatever the plugin calls them."""
    if solver == "ccopt":
        return sum(s["ccopt_stats"]["iter_count"] for s in stats)
    # One outer homotopy iteration per sigma, each an IPOPT solve of its own.
    return sum(s["iter_count"] for st in stats for s in st["nlp_stats"])


def _simulate_once(case, n_fe):
    """
    One run of the operating point on a freshly built integrator.

    A second `simulate` on the same integrator would warm start from the previous solution, and for
    both plugins that is markedly *worse* than the cold start (the homotopy restarts at `sigma_0`
    from an almost complementary point), so every repeat gets its own integrator. The price is that
    the wall time around `simulate` also contains building the solver; the plugins' own
    `wall_time_total` below is the build free number.
    """
    discretization, eps_cls, solver = case
    opts = ceiling.get_default_options(
        MIN_H_FRACTION, n_fe,
        cls_discretization=getattr(nosnoc.ClsDiscretization, discretization),
        eps_cls=eps_cls)
    integrator = nosnoc.Integrator(ceiling.get_ceiling_model(), opts,
                                   get_integrator_options(solver))
    t_start = time.perf_counter()
    t_grid, x_res, _, _ = integrator.simulate(X0)
    return integrator, t_grid, x_res, time.perf_counter() - t_start


def run_case(case, n_fe):
    """
    Solve the operating point with one case (see `CASES`) and one number of finite elements, and
    return everything the table and the plot need. Called in the child process, so it may only
    return JSON serialisable data.
    """
    discretization, eps_cls, solver = case
    _simulate_once(case, n_fe)  # warm-up, discarded

    t_solve, t_end_to_end = [], []
    for _ in range(N_REPEATS):
        integrator, t_grid, x_res, t_wall = _simulate_once(case, n_fe)
        t_end_to_end.append(t_wall)
        # The plugins' own measure, which excludes building the solver: the sum of the IPOPT wall
        # times over the homotopy loop for `reg_homotopy`, the wall time around the single MadNLP
        # call for `ccopt`.
        t_solve.append(sum(s["wall_time_total"] for s in integrator.plugin.stats))

    stats = integrator.plugin.stats
    # `converged` is not comparable across the plugins: `reg_homotopy` reports an IPOPT status plus
    # `comp_res <= complementarity_tol`, `ccopt` reports MadNLP's `success`, which is False as soon
    # as the iteration cap is hit even when the residuals are small. `constraint_violation` is the
    # number to compare, both plugins define it as max(complementarity residual, primal residual).
    return {
        "discretization": discretization,
        "eps_cls": eps_cls,
        "solver": solver,
        "n_fe": n_fe,
        "converged": bool(all(s["converged"] for s in stats)),
        "constraint_violation": float(max(s["constraint_violation"] for s in stats)),
        "n_iter": int(_solver_iterations(solver, stats)),
        "t_solve": t_solve,
        "t_end_to_end": t_end_to_end,
        "t_grid": np.asarray(t_grid).tolist(),
        "x_res": np.asarray(x_res).tolist(),
        "h": integrator.get("h").reshape(N_SIM, n_fe).tolist(),
        "lambda_max": ceiling.lambda_max(integrator),
    }


def velocity_errors(res, t_a, v_a, t_imp_a):
    """
    Deviation of the simulated velocity from the analytic one.

    `Integrator.simulate` reports, for every finite element, both boundary states of a CLS step: the
    post impact state at the left boundary and the pre impact state at the right boundary. With
    `no_initial_impacts` the first element contributes only its right boundary, so the rows of
    `x_res` are `x0`, then `rbp(0)`, then `lbp(jj), rbp(jj)` for `jj >= 1`, i.e.

        lbp(jj) = 2*jj,   rbp(jj) = 1 + 2*jj.

    Where the velocity drops inside that layout depends on the discretization: FESD-J jumps at an
    element boundary, `rbp(jj) -> lbp(jj+1)`, the relaxed OC drops across one element,
    `lbp(jj) -> rbp(jj)`. Either way it is a drop between two *consecutive* rows.

    It is not, however, simply the *largest* such drop: between two rows a time `dt` apart free
    flight already costs `g*dt`, and at coarse `N_fe` a free fall step over a long element beats the
    impact (at `N_fe = 5` the impact is 2.32 m/s against 2.4 m/s of free fall). The impact is the
    drop that gravity does not account for, so it is found on `dv + g*dt` instead, which is ~0 in
    free flight and strongly negative only across the impact. This also handles the impulsive case,
    where the two rows share a time and `dt` is 0.

    That is also why the contact element is not taken as `argmin(h)`: under the relaxed OC `ccopt`
    collapses a whole run of elements around the switch (measured
    `h = [..., 1.2e-11, 2.4e-08, 4.7e-06, ...]`) and the impact sits on the middle one, so the
    shortest element would yield a pre impact boundary.
    """
    t_grid = np.asarray(res["t_grid"])
    v = np.asarray(res["x_res"])[:, 1]
    h = np.asarray(res["h"])

    # Velocity change in excess of what free flight explains; see the docstring.
    excess = np.diff(v) + ceiling.GRAVITY*np.diff(t_grid)
    i_pre = int(np.argmin(excess))  # last pre impact row
    i_post = i_pre + 1              # first post impact row
    # Inverts the row layout above: the element across which (relaxed OC) or at whose right boundary
    # (FESD-J) the velocity drops.
    j_contact = i_pre//2
    t_imp_num = float(t_grid[i_pre])

    # The analytic velocity jumps at t_imp_a, so a pointwise comparison is only meaningful away from
    # the interval between the analytic and the simulated impact. Split there instead of picking an
    # arbitrary window.
    lo, hi = min(t_imp_num, t_imp_a), max(t_imp_num, t_imp_a)
    off_impact = (t_grid < lo) | (t_grid > hi)
    dev = np.abs(v[off_impact] - np.interp(t_grid[off_impact], t_a, v_a))

    return {
        "v_err_T": float(abs(v[-1] - np.interp(t_grid[-1], t_a, v_a))),
        "v_post_impact": float(v[i_post]),   # 0 for a plastic impact
        "v_err_max_off_impact": float(np.max(dev)) if dev.size else float("nan"),
        "t_imp_num": t_imp_num,
        "t_imp_err": float(abs(t_imp_num - t_imp_a)),
        "h_contact": float(h.flatten()[j_contact]),
        "h_min": float(h.min()),
    }


def case_label(case_or_res):
    """
    Short name of a case, spelling out which pieces of FESD-J's boundary block are switched on.

    `+xc` is the `lambda_normal perp Y_gap` cross complementarity, `+eps` the Eq. (18)
    non-penetration constraint. Takes either a `CASES` tuple or a result dict.
    """
    if isinstance(case_or_res, dict):
        case_or_res = (case_or_res["discretization"], case_or_res.get("eps_cls"),
                       case_or_res["solver"])
    discretization, eps_cls, solver = case_or_res
    suffix = "+eps" if eps_cls else ""
    return f"{discretization.lower()}{suffix}/{solver}"


def _case_arg(case, n_fe):
    """Encode a case for the child process' `--case`, inverted by `_parse_case_arg`."""
    discretization, eps_cls, solver = case
    return f"{discretization}:{eps_cls!r}:{solver}:{n_fe}"


def _parse_case_arg(arg):
    """Inverse of `_case_arg`, returning `(case, n_fe)`."""
    discretization, eps_cls, solver, n_fe = arg.split(":")
    return (discretization, float(eps_cls), solver), int(n_fe)


def run_case_in_subprocess(case, n_fe):
    """Run one case in a fresh interpreter, see the module docstring for why."""
    discretization, eps_cls, solver = case
    proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                           "--case", _case_arg(case, n_fe)],
                          capture_output=True, text=True, env=os.environ.copy(),
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])

    # A case that blows up, or a missing ccopt build, must not take the rest of the sweep down.
    failed = {"discretization": discretization, "eps_cls": eps_cls, "solver": solver,
              "n_fe": n_fe, "failed": True, "returncode": proc.returncode}
    print(f"\n{case_label(failed)}: the case failed (exit code {proc.returncode}), "
          f"last lines of stderr:")
    for line in proc.stderr.splitlines()[-20:]:
        print(f"  | {line}")
    return failed


def print_comparison_table(results, n_fe, t_imp_a, v_pre_a):
    """
    One table per N_fe, one row per case.

    `t solve` is the plugin's own measure and excludes building the solver, `t e2e` wraps the whole
    `Integrator.simulate` call on a fresh integrator and therefore includes the build; both are
    reported as `min (median)` over `N_REPEATS` repeats. `cv` is the constraint violation, which
    unlike `converged` means the same thing for both plugins. `h contact` is the element carrying
    the impact and `h min` the shortest element overall; the two differ only where a solver
    collapses several elements around the switch.
    """
    h0 = ceiling.h0_for(n_fe)
    print(f"\nN_fe = {n_fe}: h_0 = {h0:.5e} s, "
          f"h floor = {MIN_H_FRACTION:g}*h_0 = {MIN_H_FRACTION*h0:.5e} s")
    print(f"analytic impact at t = {t_imp_a:.6f} s with v = {v_pre_a:.6f} m/s, "
          f"best of {N_REPEATS} repeats")
    header = (f"{'case':>34} {'conv':>5} {'cv':>10} {'t solve [s]':>19} {'t e2e [s]':>19} "
              f"{'iter':>7} {'h contact':>10} {'h min':>10} {'v err(T)':>10} {'v post':>10} "
              f"{'v err max':>10} {'t imp err':>10}")
    print(header)
    print("-"*len(header))
    for res in results:
        if res.get("failed"):
            print(f"{case_label(res):>34} {'FAILED, see stderr above':>26}")
            continue
        err = res["errors"]
        ts = f"{min(res['t_solve']):>8.3f} ({median(res['t_solve']):>7.3f})"
        te = f"{min(res['t_end_to_end']):>8.3f} ({median(res['t_end_to_end']):>7.3f})"
        print(f"{case_label(res):>34} {str(res['converged']):>5} "
              f"{res['constraint_violation']:>10.2e} "
              f"{ts:>19} {te:>19} {res['n_iter']:>7} {err['h_contact']:>10.2e} "
              f"{err['h_min']:>10.2e} {err['v_err_T']:>10.2e} {err['v_post_impact']:>10.2e} "
              f"{err['v_err_max_off_impact']:>10.2e} {err['t_imp_err']:>10.2e}")


def plot_comparison(results, t_a, v_a, t_imp_a, v_pre_a):
    """
    The sweep as a work-precision picture, one line per case, plus the trajectories at the finest
    N_fe so there is still something physical to look at.
    """
    nosnoc.latexify_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.7, len(CASES)))
    n_fe_last = max(N_FE_SWEEP)

    axes[2].plot(t_a, v_a, "k--", linewidth=1.2, label="analytic")

    for case, color in zip(CASES, colors):
        label = case_label(case)
        done = [res for res in results
                if not res.get("failed") and case_label(res) == label]
        if not done:
            continue
        n_fe = [res["n_fe"] for res in done]
        axes[0].semilogy(n_fe, [res["errors"]["v_err_T"] for res in done], "-o",
                         markersize=3.5, color=color, label=label)
        axes[1].semilogy(n_fe, [min(res["t_solve"]) for res in done], "-o",
                         markersize=3.5, color=color, label=label)
        for res in done:
            if res["n_fe"] == n_fe_last:
                axes[2].plot(np.asarray(res["t_grid"]), np.asarray(res["x_res"])[:, 1], "-o",
                             markersize=2.5, color=color, label=label)

    axes[0].set_xlabel("$N_\\mathrm{fe}$")
    axes[0].set_ylabel(r"$|v(T) - v_\mathrm{analytic}(T)|$")
    axes[0].set_title("terminal velocity error")
    axes[0].grid()
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("$N_\\mathrm{fe}$")
    axes[1].set_ylabel("$t$ [s]")
    axes[1].set_title("time in the solver")
    axes[1].grid()
    axes[1].legend(fontsize=8)

    axes[2].axvline(t_imp_a, color="k", linewidth=0.6)
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel("$v$")
    axes[2].set_title(f"velocity, $N_\\mathrm{{fe}} = {n_fe_last}$")
    axes[2].grid()
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def example(plot=True):
    t_a, q_a, v_a, t_imp_a, v_pre_a = ceiling.analytic_solution()

    results = []
    for n_fe in N_FE_SWEEP:
        per_n_fe = []
        for case in CASES:
            print(f"running {case_label(case)}, N_fe = {n_fe} ...")
            res = run_case_in_subprocess(case, n_fe)
            if not res.get("failed"):
                res["errors"] = velocity_errors(res, t_a, v_a, t_imp_a)
            per_n_fe.append(res)
        print_comparison_table(per_n_fe, n_fe, t_imp_a, v_pre_a)
        results.extend(per_n_fe)

    if plot:
        plot_comparison(results, t_a, v_a, t_imp_a, v_pre_a)

    return results


def main(argv):
    if "--case" in argv:
        case, n_fe = _parse_case_arg(argv[argv.index("--case") + 1])
        print(RESULT_SENTINEL + json.dumps(run_case(case, n_fe)))
        return
    example(plot="--no-plot" not in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
