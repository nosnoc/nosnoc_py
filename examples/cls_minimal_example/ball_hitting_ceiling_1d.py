
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import nosnoc

GRAVITY = 9.81
CEILING = 1.0

X0 = np.array([0.0, 5.0])  # start on the ground, thrown upwards
T_SIM = 1
N_SIM = 1
N_FE = 100

H_SIM = T_SIM/N_SIM  # length of one integrator step, i.e. of the single control stage
H0 = H_SIM/N_FE      # nominal finite element length, the floor is a fraction of this

# Floors on the finite element length, as a fraction of the nominal step h_0, in decreasing order.
# `gamma_h_lb` is the fraction by which h may *shrink*, so a floor of f*h_0 is gamma_h_lb = 1 - f.
MIN_H_FRACTIONS = [1e-3,1e-4,1e-5]

# Ceiling on the finite element length, h <= (1 + GAMMA_H_UB)*h_0. Kept fixed over the sweep.
GAMMA_H_UB = 1.0



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


def get_default_options(min_h_fraction, **kwargs):
    default_args = {
        "N_stages": 1,
        "N_finite_elements": N_FE,
        "n_s": 3,
        "rk_scheme": nosnoc.RKScheme.RADAU_IIA,
        "use_fesd": True,
        "cls_discretization": nosnoc.ClsDiscretization.RELAXED_OC,
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


def get_default_integrator_options(**kwargs):
    solver_opts = nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopyOptions()

    solver_opts.homotopy_update_slope = 0.2
    solver_opts.N_homotopy = 15
    # The relaxed OC satisfies the complementarity only up to O(h) on the contact element, so the
    # homotopy cannot be driven as far as it can for FESD-J.
    solver_opts.complementarity_tol = 1e-6
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


def solve_ceiling(min_h_fraction, x0=X0):
    """Simulate one sweep point and return the trajectory plus the quantities set by the floor."""
    opts = get_default_options(min_h_fraction)
    model = get_ceiling_model(x0=x0)
    integrator = nosnoc.Integrator(model, opts, get_default_integrator_options())
    t_grid, x_res, _, _ = integrator.simulate(x0)

    # One row per integrator step, one column per finite element. `get_full` is needed for the
    # contact force, whose peak sits at an interior collocation point; `get` only reports the
    # element end points, where it is typically already back to zero.
    h = integrator.get("h").reshape(N_SIM, N_FE)
    

    return {
        "min_h_fraction": min_h_fraction,
        "h_floor": min_h_fraction*H0,
        "t_grid": t_grid,
        "x_res": x_res,
        "h": h,
        "h_min_per_step": h.min(axis=1),
        "h_min": h.min(),
        "lambda_max": np.max(integrator.get_full("lambda_normal")),
    }



def print_h_table(results):
    """
    Finite element lengths against the floor imposed on them.

    One column per integrator step, holding the shortest element the solver used in that step. The
    floor is the same in every step, so the table shows *where* it is attained: the step containing
    the impact puts an element on it, the others should stay near h_0. The last column is the peak
    contact force, which grows like 1/h as the floor is lowered.
    """
    print(f"h_0 = {H0:.5e} s, upper bound (1+gamma_h_ub)*h_0 = {(1+GAMMA_H_UB)*H0:.5e} s")
    steps = " ".join(f"{'step ' + str(i+1):>10}" for i in range(N_SIM))
    header = (f"{'fraction':>9} {'h floor':>11} | {steps} | {'min h':>11} {'at floor':>9} "
              f"{'lambda max':>11} {'v error':>10}")
    print(header)
    print("-"*len(header))
    for res in results:
        mins = " ".join(f"{m:>10.3e}" for m in res["h_min_per_step"])
        at_floor = "yes" if res["h_min"] <= res["h_floor"]*1.001 else "no"
        print(f"{res['min_h_fraction']:>9g} {res['h_floor']:>11.3e} | {mins} | "
              f"{res['h_min']:>11.3e} {at_floor:>9} {res['lambda_max']:>11.3e} "
              f"{res['v_error']:>10.2e}")


def plot_results(results):
    nosnoc.latexify_plot()
    t_a, q_a, v_a, t_imp, v_pre = analytic_solution()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for ax in axes:
        ax.plot(t_a, v_a, "k--", linewidth=1.2, label="analytic")
    for res, color in zip(results, colors):
        label = rf"$h \geq {res['min_h_fraction']:g}\,h_0$"
        for ax in axes:
            ax.plot(res["t_grid"], res["x_res"][:, 1], "-o", markersize=2.5,
                    color=color, label=label)

    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$v$")
    axes[0].set_title("velocity")
    axes[0].grid()
    axes[0].legend(fontsize=8)

    # Zoom on the impact: this is where the floor on h is visible as a plateau at v = 0.
    span = 3*MIN_H_FRACTIONS[0]*H0
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
    for min_h_fraction in MIN_H_FRACTIONS:
        res = solve_ceiling(min_h_fraction)
        res["v_error"] = abs(np.interp(res["t_grid"][-1], t_a, v_a) - res["x_res"][-1, 1])
        results.append(res)

    print_h_table(results)
    

    if plot:
        plot_results(results)
    
    return results


if __name__ == "__main__":
    example()