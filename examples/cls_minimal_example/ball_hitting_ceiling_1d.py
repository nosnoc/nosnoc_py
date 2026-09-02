
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 9.81
CEILING = 1.0

X0 = np.array([0.0, 5.0])  # start on the ground, thrown upwards
T_SIM = 1
N_SIM = 1
# Sweep over the number of finite elements per integrator step. Every value gets its own row in the
# table, and its own nominal element length h_0 = h_sim/N_fe, so the floor below moves with it.
N_FE_SWEEP = tuple(range(3, 11))

H_SIM = T_SIM/N_SIM  # length of one integrator step, i.e. of the single control stage

# Floors on the finite element length, as a fraction of the nominal step h_0, in decreasing order.
# `gamma_h_lb` is the fraction by which h may *shrink*, so a floor of f*h_0 is gamma_h_lb = 1 - f.
MIN_H_FRACTIONS = [1.0e-10]

# Ceiling on the finite element length, h <= (1 + GAMMA_H_UB)*h_0. Kept fixed over the sweep.
GAMMA_H_UB = 1.0

# MPCC solver to run the sweep with, either "reg_homotopy" or "ccopt". `ccopt` needs the libMad
# based casadi build, see the export block at the end of `env_ccopt/bin/activate`.
MPCC_SOLVER = "ccopt"  # "ccopt" or "reg_homotopy"

# Print the decision variables of the solver's last iteration for every sweep point, see
# `print_last_iterate`. Works for both MPCC solvers.
PRINT_LAST_ITERATE = True



def get_ceiling_model(x0=X0):
    """Build the ball-and-ceiling system as a `nosnoc.model.Cls`."""
    q = ca.SX.sym("q")
    v = ca.SX.sym("v")
    return nosnoc.model.Cls(
        x=ca.vertcat(q, v),
        x0=x0,
        M=np.eye(1),
        f_v=ca.SX(-GRAVITY),
        f_c=CEILING - q,  # gap function, the ball touches the ceiling at q = CEILING
        e=0.0,            # the relaxed OC can only represent a plastic impact, see below
        mu=0.0,           # frictionless
        name="ball_hitting_ceiling",
    )


def h0_for(n_fe):
    """Nominal finite element length for a step split into `n_fe` elements."""
    return H_SIM/n_fe


def lambda_max(integrator):
    """
    Peak contact force over the whole trajectory, or 0 where the discretization has none.

    `RELAXED_OC_IMPULSE_ONLY` carries no `lambda_normal` variable at all, and `get_full` raises on a
    variable that was never created, so it is reported as an exact zero: with that discretization
    the contact really does exert no force, only impulses.
    """
    if (integrator.plugin.opts.cls_discretization
            == nosnoc.ClsDiscretization.RELAXED_OC_IMPULSE_ONLY):
        return 0.0
    return float(np.max(integrator.get_full("lambda_normal")))


def impulses(integrator, n_fe):
    """
    Contact impulses per finite element, and the time at which each one acts.

    `Lambda_normal` lives at the *left boundary* of a finite element, and with
    `no_initial_impacts` it is not defined on the first element of a step, so `get` returns
    `N_sim*(n_fe-1)` values: row `r` of a step belongs to element `r+2`, acting at the boundary
    `sum(h[:r+1])` into that step.

    Returns `(Lambda, t_boundary)`, both shaped `(N_sim, n_fe-1)`, or `(None, None)` for a
    discretization that has no impulses at all.
    """
    if not integrator.plugin.dtp._has_impulse():
        return None, None
    Lambda = np.asarray(integrator.get("Lambda_normal")).reshape(N_SIM, n_fe-1)
    h = np.asarray(integrator.get("h")).reshape(N_SIM, n_fe)
    # Absolute time of each element's left boundary: where the step starts, plus the elements
    # already consumed inside it. Element 1 is dropped to line up with Lambda.
    step_start = np.concatenate([[0.0], np.cumsum(h.sum(axis=1))[:-1]])
    t_boundary = step_start[:, None] + np.cumsum(h, axis=1)[:, :n_fe-1]
    return Lambda, t_boundary


def get_default_options(min_h_fraction, n_fe, **kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": n_fe,
        "n_s": 3,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": True,
        # Contact only ever as an impulse: no `lambda_normal` at all, so the dynamics are free
        # flight between the element boundaries and the *smeared* impact that RELAXED_OC and
        # RELAXED_OC_IMPULSE both admit -- a large contact force over one collapsing element -- has
        # no variable left to live in. Switch to RELAXED_OC_IMPULSE to put the force back while
        # keeping the impulse, or to FESD_J to forbid the smeared impact by constraint instead.
        "cls_discretization": nosnoc.ClsDiscretization.RELAXED_OC_IMPULSE_ONLY,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_FE,
        "no_initial_impacts": True,
        "gamma_h_ub": GAMMA_H_UB,  # upper bound on the finite element length, h <= (1+gamma_h_ub)*h_0
        # Lower bound h >= (1-gamma_h_lb)*h_0, i.e. h >= min_h_fraction*h_0.
        "gamma_h_lb": 1.0 - min_h_fraction,
        "step_equilibration": nosnoc.StepEquilibrationMode.L2_RELAXED_SCALED,
        "rho_h": 0.0, # no step equilibration, h stay free in [(1-gamma_h_lb)*h0, (1+gamma_h_ub)*h0]
        "initial_Y_gap": 0.0,
        "initial_y_gap": 0.0,
        # `Options` wants exactly one of T, h, h_k. With N_stages = 1 the control stage is one
        # integrator step; `Integrator` overwrites this with h_sim = T_SIM/N_SIM anyway.
        "T": H_SIM,
    }
    return nosnoc.Options(**(default_args | kwargs))


def get_default_solver_options():
    """Options for the MPCC solver picked by `MPCC_SOLVER`."""
    if MPCC_SOLVER == "ccopt":
        solver_opts = nosnoc.mpccsol.plugins.ccopt.CCOptOptions()
        solver_opts.madnlp_opts["linear_solver"] = "Ma27Solver"
        
        


    else: 
        solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()
        solver_opts.opts_casadi_nlp["ipopt"]["linear_solver"] = "ma27"

    
    

    solver_opts.homotopy_update_slope = 0.2
    solver_opts.N_homotopy = 15
    # The relaxed OC satisfies the complementarity only up to O(h) on the contact element, so the
    # homotopy cannot be driven as far as it can for FESD-J.
    solver_opts.complementarity_tol = 1e-6
    return solver_opts


def get_default_integrator_options(**kwargs):
    solver_opts = get_default_solver_options()
    default_args = {
        "T_sim": T_SIM,
        "N_sim": N_SIM,
        "solver_opts": solver_opts,
        "print_level": 5,
    }
    return nosnoc.FESDIntegratorOptions(**(default_args | kwargs))


def analytic_solution(x0=X0, t_sim=T_SIM, n_points=2000):
    """
    Analytic trajectory: free flight up, plastic impact on the ceiling, free fall back down.

    Returns the time grid, the positions, the velocities, the impact time and the impulse
    Lambda_n = M |v(t_s^-)| needed to bring the ball to rest.
    """
    q0, v0 = x0
    disc = v0**2 - 2*GRAVITY*(CEILING - q0)
    if disc <= 0:
        raise RuntimeError("The ball does not reach the ceiling, increase v0.")
    t_imp = (v0 - np.sqrt(disc))/GRAVITY
    v_pre = np.sqrt(disc)

    t_up = np.linspace(0.0, t_imp, int(n_points*t_imp/t_sim))
    q_up = q0 + v0*t_up - 0.5*GRAVITY*t_up**2
    v_up = v0 - GRAVITY*t_up

    t_down = np.linspace(t_imp, t_sim, int(n_points*(t_sim - t_imp)/t_sim))
    tt = t_down - t_imp
    q_down = CEILING - 0.5*GRAVITY*tt**2
    v_down = -GRAVITY*tt

    return (np.concatenate([t_up, t_down]), np.concatenate([q_up, q_down]),
            np.concatenate([v_up, v_down]), t_imp, v_pre)


def solve_ceiling(n_fe, min_h_fraction, x0=X0):
    """Simulate one sweep point and return the trajectory plus the quantities set by the floor."""
    opts = get_default_options(min_h_fraction, n_fe)
    model = get_ceiling_model(x0=x0)
    integrator = nosnoc.Integrator(model, opts, get_default_integrator_options())
    if MPCC_SOLVER == "reg_homotopy":  # the dump hooks into the homotopy loop, ccopt has none
        dump_at_homotopy_iters(integrator, iters=(0,))
    t_grid, x_res, _, _ = integrator.simulate(x0)

    if PRINT_LAST_ITERATE:
        print_last_iterate(integrator, n_fe, min_h_fraction)


    # One row per integrator step, one column per finite element. `get_full` is needed for the
    # contact force, whose peak sits at an interior collocation point; `get` only reports the
    # element end points, where it is typically already back to zero.
    h = integrator.get("h").reshape(N_SIM, n_fe)
    Lambda, t_boundary = impulses(integrator, n_fe)


    return {
        "n_fe": n_fe,
        "h_0": h0_for(n_fe),
        "min_h_fraction": min_h_fraction,
        "h_floor": min_h_fraction*h0_for(n_fe),
        "t_grid": t_grid,
        "x_res": x_res,
        "h": h,
        "h_min_per_step": h.min(axis=1),
        "h_min": h.min(),
        "lambda_max": lambda_max(integrator),
        "Lambda": Lambda,
        "t_boundary": t_boundary,
    }



def print_h_table(results):
    """
    Finite element lengths against the floor imposed on them, one row per sweep point.

    The sweep runs over the number of finite elements, so `h_0 = h_sim/N_fe` and with it the floor
    `fraction*h_0` change from row to row. One column per integrator step, holding the shortest
    element the solver used in that step. The floor is the same in every step, so the table shows
    *where* it is attained: the step containing the impact puts an element on it, the others should
    stay near h_0. `lambda max` is the peak contact force, which grows like 1/h as the floor is
    lowered.
    """
    print(f"h_sim = {H_SIM:.5e} s, h_0 = h_sim/N_fe, "
          f"h in [fraction*h_0, {1+GAMMA_H_UB:g}*h_0]")
    steps = " ".join(f"{'step ' + str(i+1):>10}" for i in range(N_SIM))
    header = (f"{'N_fe':>5} {'h_0':>11} {'fraction':>9} {'h floor':>11} | {steps} | "
              f"{'min h':>11} {'at floor':>9} {'lambda max':>11} {'v error':>10}")
    print(header)
    print("-"*len(header))
    for res in results:
        mins = " ".join(f"{m:>10.3e}" for m in res["h_min_per_step"])
        at_floor = "yes" if res["h_min"] <= res["h_floor"]*1.001 else "no"
        print(f"{res['n_fe']:>5d} {res['h_0']:>11.3e} {res['min_h_fraction']:>9g} "
              f"{res['h_floor']:>11.3e} | {mins} | "
              f"{res['h_min']:>11.3e} {at_floor:>9} {res['lambda_max']:>11.3e} "
              f"{res['v_error']:>10.2e}")


def print_impulse_table(results):
    """
    The contact impulse on every finite element, one block per sweep point.

    A correct plastic impact puts the whole momentum change into a *single* element boundary,
    Lambda_n = M|v(t_s^-)|, and leaves every other boundary at zero. So the row should read as
    noise everywhere except one entry, marked `*`, and that entry should sit at the analytic impact
    time. A row with two comparable entries means the solver split the impact over two boundaries;
    a row that is noise throughout means no impact was detected at all.
    """
    _, _, _, t_imp_a, v_pre_a = analytic_solution()
    print(f"\ncontact impulses per finite element, analytic Lambda_n = {v_pre_a:.6f} Ns "
          f"at t = {t_imp_a:.6f} s")
    print("element 1 carries no impulse (no_initial_impacts), shown as '--'")

    for res in results:
        Lambda, t_boundary = res["Lambda"], res["t_boundary"]
        if Lambda is None:
            print(f"\n  N_fe = {res['n_fe']}: this discretization has no impulse variables")
            continue
        for step in range(N_SIM):
            L, tb = Lambda[step], t_boundary[step]
            i_max = int(np.argmax(L))
            head = f"  N_fe = {res['n_fe']:>2}" + (f", step {step+1}" if N_SIM > 1 else "")
            print(f"\n{head}: impact at t = {tb[i_max]:.6f} s "
                  f"(err {abs(tb[i_max] - t_imp_a):.1e}), "
                  f"Lambda_n = {L[i_max]:.6f} (err {abs(L[i_max] - v_pre_a):.1e})")
            print("    element " + "".join(f"{jj:>12}" for jj in range(1, res['n_fe']+1)))
            print("    t       " + f"{0.0:>12.6f}" + "".join(f"{t:>12.6f}" for t in tb))
            print("    Lambda_n" + f"{'--':>12}"
                  + "".join(f"{v:>11.3e}" + ("*" if r == i_max else " ")
                            for r, v in enumerate(L)))


def plot_results(results):
    nosnoc.latexify_plot()
    t_a, q_a, v_a, t_imp, v_pre = analytic_solution()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for ax in axes:
        ax.plot(t_a, v_a, "k--", linewidth=1.2, label="analytic")
    for res, color in zip(results, colors):
        label = rf"$N_\mathrm{{fe}} = {res['n_fe']}$, $h \geq {res['min_h_fraction']:g}\,h_0$"
        for ax in axes:
            ax.plot(res["t_grid"], res["x_res"][:, 1], "-o", markersize=2.5,
                    color=color, label=label)

    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$v$")
    axes[0].set_title("velocity")
    axes[0].grid()
    axes[0].legend(fontsize=8)

    # Zoom on the impact: this is where the floor on h is visible as a plateau at v = 0. The window
    # follows the coarsest sweep point, whose contact element is the longest.
    span = 3*max(res["h_floor"] for res in results)
    axes[1].set_xlim(t_imp - span, t_imp + 2*span)
    axes[1].set_ylim(-1.2, 1.2*v_pre)
    axes[1].axvline(t_imp, color="k", linewidth=0.6)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("$v$")
    axes[1].set_title("impact, zoomed: the plateau is the contact element")
    axes[1].grid()

    plt.tight_layout()
    plt.show()


def example(plot=True):
    t_a, q_a, v_a, t_imp, v_pre = analytic_solution()
    print(f"analytic impact at t = {t_imp:.6f} s with v = {v_pre:.6f} m/s, "
          f"impulse = {v_pre:.6f} Ns")

    results = []
    for n_fe in N_FE_SWEEP:
        for min_h_fraction in MIN_H_FRACTIONS:
            res = solve_ceiling(n_fe, min_h_fraction)
            res["v_error"] = abs(np.interp(res["t_grid"][-1], t_a, v_a) - res["x_res"][-1, 1])
            results.append(res)

    print_h_table(results)
    print_impulse_table(results)


    if plot:
        plot_results(results)
    
    return results


def print_last_iterate(integrator, n_fe, min_h_fraction):
    """
    Print the decision variables of the solver's last iteration.

    `Mpcc.solve` copies the final primal-dual iterate back into the dtp, so after `simulate` the dtp
    already holds it: value (`res`), bounds, multiplier and bound violation per variable. Unlike
    `dump_at_homotopy_iters` this needs no hook into the homotopy loop and therefore works for
    `ccopt` as well. With N_SIM > 1 only the last integrator step survives, since every step reuses
    the same dtp.

    One row reads confusingly: `simulate` advances the bounds on `x_0` to the terminal state at the
    end of every step, ready for the next one, so for `x_0` the `lb`/`ub`/`init` columns show where
    the *next* step would start while `res` still holds the state the solve actually used.

    `print(dtp)` instead of the two blocks below also dumps the objective, the parameters and the
    constraints g.
    """
    dtp = integrator.plugin.dtp
    print(f"\ndecision variables at the last iteration, N_fe = {n_fe}, "
          f"h >= {min_h_fraction:g}*h_0:")
    print(dtp.w)
    print_complementarities(dtp)


def print_complementarities(dtp):
    """
    Print the complementarity pairs 0 <= G_i perp H_i >= 0 of the last iteration.

    `print(dtp.G)` and `print(dtp.H)` render two separate tables, which hides the only thing that
    matters here: which *pair* is still off. The two line up index by index, so they are joined into
    one row each, with the product that should be zero. The worst pair is marked with a `*`; that is
    the residual the homotopy (or the ccopt relaxation) is driving down.
    """
    G = np.asarray(dtp.G.val).flatten()
    H = np.asarray(dtp.H.val).flatten()
    prod = G*H
    i_worst = int(np.argmax(np.abs(prod)))

    print(f"complementarities, max |G*H| = {abs(prod[i_worst]):.3e}:")
    header = f"{'#':>4} {'G':>21} {'G val':>12} {'H':>21} {'H val':>12} {'G*H':>12}"
    print(header)
    print("-"*len(header))
    for i in range(len(G)):
        print(f"{i:>4d} {str(dtp.G.sym[i]):>21} {G[i]:>12.4e} "
              f"{str(dtp.H.sym[i]):>21} {H[i]:>12.4e} {prod[i]:>12.3e}"
              f"{' *' if i == i_worst else ''}")


def dump_at_homotopy_iters(integrator, iters=(2,)):
    """Print the dtp (vars, bounds, residuals) after the given reg_homotopy iterations."""
    dtp = integrator.plugin.dtp
    if dtp.solver is None:            # solver is built lazily on the first solve
        dtp.create_solver(integrator.plugin.integrator_opts.solver_opts,
                          plugin="reg_homotopy")
    pl = dtp.solver
    orig = pl._solve_nlp
    k = {"i": 0}

    def patched():
        orig()
        k["i"] += 1
        if k["i"] in iters:
            w, p = pl.nlp.w.res, pl.nlp.p.val
            dtp.w.res  = pl.w_mpcc_fun(w).full().flatten()
            dtp.w.mult = pl.w_mpcc_fun(pl.nlp.w.mult).full().flatten()
            dtp.g.val  = pl.g_mpcc_fun(w, p).full().flatten()
            dtp.g.mult = pl.nlp.g.mult[pl.ind_g_mpcc]
            dtp.G.val  = pl.G_mpcc_fun(w, p).full().flatten()
            dtp.H.val  = pl.H_mpcc_fun(w, p).full().flatten()
            st = pl.stats["nlp_stats"][-1]
            print(f"\nHOMOTOPY ITER {k['i']}  sigma={pl._sigma_curr():.3e}  "
                    f"status={st['return_status']}  inf_pr={st['iterations']['inf_pr'][-1]:.3e}")
            print(dtp)

    pl._solve_nlp = patched
    return pl



if __name__ == "__main__":
    example()