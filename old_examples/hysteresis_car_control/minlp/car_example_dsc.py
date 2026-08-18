"""Create MINLP for the hysteresis car problem."""

from typing import Union, List, Optional, Dict
from nosnoc.rk_utils import generate_butcher_tableu_integral, IrkSchemes
import casadi as ca
from casadi import MX, SX, vertcat, inf
import numpy as np
import pickle
from sys import argv
from time import perf_counter


def tic():
    """Tic."""
    global perf_ti
    perf_ti = perf_counter()


def toc():
    """Toc."""
    global perf_ti
    tim = perf_counter()
    dt = tim - perf_ti
    print("  Elapsed time: %s s." % (dt))
    perf_ti = tim
    return dt


def make_list(value, nr=1):
    """Make list."""
    if not isinstance(value, list):
        return [value] * nr
    else:
        return value


class Description:
    """Description for Casadi."""

    def __init__(self, variable_type=SX):
        """Create description."""
        self.g = []
        self.ubg = []
        self.lbg = []
        self.w = []
        self.w0 = []
        self.indices = {}  # Names with indices
        self.p = []
        self.p0 = []
        self.indices_p = {}  # Names with indices
        self.lbw = []
        self.ubw = []
        self.discrete = []
        self.f = 0
        self.solver = None
        self.solution = None
        self.variable_type = variable_type

        # For big M reformulations:
        self.M = 1e4
        self.eps = 1e-9

    def add_g(self, mini: float, equation: Union[SX, MX], maxi: float) -> int:
        """
        Add to g.

        :param mini: minimum
        :param equation: constraint equation
        :param maxi: maximum
        :return: index of constraint
        """
        nr = equation.shape[0] * equation.shape[1]
        self.lbg += make_list(mini, nr)
        self.g += make_list(equation)
        self.ubg += make_list(maxi, nr)
        return len(self.ubg) - 1

    def leq(self, op1, op2) -> int:
        """Lower or equal."""
        if isinstance(op1, (float, int, list)):
            op1 = make_list(op1)
            nr = len(op1)
            return self.add_g(op1, op2, [inf] * nr)
        elif isinstance(op2, (float, int, list)):
            op2 = make_list(op2)
            nr = len(op2)
            return self.add_g([-inf] * nr, op1, op2)
        else:
            diff = op1 - op2
            assert (diff.shape[1] == 1)
            nr = diff.shape[0]
            return self.add_g([-inf] * nr, diff, [0] * nr)

    def eq(self, op1, op2) -> int:
        """Equal."""
        diff = op1 - op2
        assert (diff.shape[1] == 1)
        nr = diff.shape[0]
        return self.add_g([0] * nr, op1 - op2, [0] * nr)

    def equal_if_on(self, trigger, equality):
        """Big M formulation."""
        diff = (1 - trigger)
        self.leq(-self.M * diff, equality)
        self.leq(equality, self.M * diff)

    def higher_if_on(self, trigger, equation):
        """Trigger = 1 if equation > -eps else lower."""
        self.leq(-(1-trigger) * self.M, equation - self.eps)
        self.leq(equation + self.eps, self.M * trigger)

    def sym_bool(
        self, name: str, nr: int = 1,
    ) -> Union[MX, SX]:
        """Create a symbolic boolean."""
        return self.sym(name, nr, 0, 1, x0=1, discrete=True)

    def sym(
        self,
        name: str,
        nr: int,
        lb: Union[float, List[float]],
        ub: Union[float, List[float]],
        x0: Optional[Union[float, List[float]]] = None,
        discrete: bool = False,
    ) -> Union[MX, SX]:
        """Create a symbolic variable."""
        # Gather Data
        if name not in self.indices:
            self.indices[name] = []
        idx_list = self.indices[name]

        # Create
        x = self.variable_type.sym("%s[%d]" % (name, len(idx_list)), nr)
        lb = make_list(lb, nr)
        ub = make_list(ub, nr)

        if x0 is None:
            x0 = lb
        x0 = make_list(x0, nr)

        if len(lb) != nr:
            raise Exception("Lower bound error!")
        if len(ub) != nr:
            breakpoint()
            raise Exception("Upper bound error!")
        if len(x0) != nr:
            raise Exception("Estimation length error (x0 %d vs nr %d)!"
                            % (len(x0), nr))

        # Collect
        out = self.add_w(lb, x, ub, x0, discrete, name=name)
        idx_list.append(out)

        return x

    def add_w_discrete(
        self,
        lb: Union[float, List[float]],
        w: Union[MX, SX],
        ub: Union[float, List[float]],
        name=None
    ) -> List:
        """Add a discrete, existing symbolic variable."""
        return self.add_w(lb, w, ub, True, name=name)

    def add_w(
        self,
        lb: Union[float, List[float]],
        w: Union[MX, SX],
        ub: Union[float, List[float]],
        x0: Optional[Union[float, List[float]]],
        discrete: bool = False,
        name=None
    ):
        """Add an existing symbolic variable."""
        idx = [i + len(self.lbw) for i in range(len(lb))]
        self.w += [w]
        self.lbw += lb
        self.ubw += ub
        self.w0 += x0
        self.discrete += [1 if discrete else 0] * len(lb)
        return idx

    def add_parameters(self, name, nr, values=0):
        """Add some parameters."""
        # Gather Data
        if name not in self.indices_p:
            self.indices_p[name] = []
        idx_list = self.indices_p[name]

        # Create
        p = self.variable_type.sym("%s[%d]" % (name, len(idx_list)), nr)
        values = make_list(values, nr)

        if len(values) != nr:
            raise Exception("Values error!")

        # Create & Collect
        new_idx = [i + len(self.p0) for i in range(nr)]
        self.p += [p]
        self.p0 += values

        self.indices_p[name].extend(new_idx)
        return p

    def set_parameters(self, name, values):
        """Set parameters."""
        idx = self.indices_p[name]
        values = make_list(values, len(idx))
        if len(idx) != len(values):
            raise Exception(
                "Idx (%d) and values (%d) should be equally long for %s!" %
                (len(idx), len(values), name)
            )

        for i, v in zip(idx, values):
            self.p0[i] = v

    def get_nlp(self) -> Dict:
        """Get nlp description."""
        res = {'x': vertcat(*(self.w)),
               'f': self.f,
               'g': vertcat(*(self.g))}
        if len(self.p0) > 0:
            res['p'] = vertcat(*(self.p))

        return res

    def get_options(self, is_discrete=True, **kwargs):
        """Get options."""
        if is_discrete and 1 in self.discrete:
            out = {'discrete': self.discrete}
            out.update(kwargs)
            return out
        return kwargs

    def set_solver(self, solver, name: str = None, options: Dict = None,
                   is_discrete=True, **kwargs):
        """
        Set the solver.

        Set the related solver using a simular line as this:
        nlpsol('solver', 'ipopt', dsc.get_nlp(), dsc.get_options())

        :param solver: solver type (for example nlpsol from casadi)
        :param name: Name of the actual solver
        :param options: Dictionary of extra options
        """
        if options is None:
            options = {}
        if name is None:
            self.solver = solver
        else:
            self.solver = solver(
                'solver', name, self.get_nlp(**kwargs),
                self.get_options(is_discrete=is_discrete, **options)
            )

    def get_solver(self):
        """Return solver."""
        return self.solver

    def get_indices(self, name: str):
        """
        Get indices of a certain variable.

        :param name: name
        """
        return self.indices[name]

    def solve(self, auto_update=False, **kwargs):
        """Solve the problem."""
        params = {
            'lbx': self.lbw,
            'ubx': self.ubw,
            'lbg': self.lbg,
            'ubg': self.ubg,
            'x0': self.w0
        }
        params.update(kwargs)
        if len(self.p0) > 0 and "p0" not in params:
            params["p"] = self.p0

        self.solution = self.solver(**params)
        self.stats = self.solver.stats()
        if auto_update:
            self.w0 = self.solution['x']
        return self.solution

    def is_solved(self):
        """Check if it is solved."""
        return self.stats['success']

    def get(self, name: str):
        """Get solution for a name."""
        if name not in self.indices:
            raise Exception(
                f"{name} not found, options {self.indices.keys()}"
            )

        if self.solution is not None:
            out = []
            for el in self.indices[name]:
                if isinstance(el, list):
                    out.append([float(self.solution['x'][i]) for i in el])
                else:
                    out.append(float(self.solution['x'][el]))

            while isinstance(out, list) and len(out) == 1:
                out = out[0]

            return out


def create_problem(time_as_parameter=False, use_big_M=False, more_stages=True, problem1=True):
    """Create problen."""
    # Parameters
    if more_stages:
        N_stages = 15 * 2
        N_finite_elements = 3
    else:
        N_stages = 15
        N_finite_elements = 3 * 2

    N_control_intervals = 1
    n_s = 2
    use_collocation = True
    B_irk, C_irk, D_irk, tau = generate_butcher_tableu_integral(
        n_s, IrkSchemes.RADAU_IIA
    )

    # Hystheresis parameters
    if problem1:
        psi_on = [10, 17.5, 25]
        psi_off = [5, 12.5, 20]
    else:
        psi_on = [10, 12.5, 15]
        psi_off = [5, 7.5, 10]

    # Model parameters
    q_goal = 300
    v_goal = 0
    v_max = 30
    u_max = 3

    # fuel costs:
    C = [1, 1.8, 2.5, 3.2]
    # ratios
    n = [1, 2, 3, 4]

    # State variables:
    q = ca.SX.sym("q")  # position
    v = ca.SX.sym("v")  # velocity
    L = ca.SX.sym("L")  # Fuel usage
    X = ca.vertcat(q, v, L)
    X0 = [0, 0, 0]
    lbx = [0, 0, -ca.inf]
    ubx = [ca.inf, v_max, ca.inf]
    n_x = 3
    # Binaries to represent the problem:
    n_y = len(n)
    Y0 = np.zeros(n_y)
    Y0[0] = 1
    u = ca.SX.sym('u')  # drive
    n_u = 1
    U0 = np.zeros(n_u)

    lbu = np.array([-u_max])
    ubu = np.array([u_max])

    # x = q v L
    X = ca.vertcat(q, v, L)

    F_dyn = [
        ca.Function(f'f_dyn_{i}', [X, u], [ca.vertcat(
            v, n[i]*u, C[i]
        )]) for i in range(len(n))
    ]

    psi = v
    psi_fun = ca.Function('psi', [X], [psi])

    # Create problem:
    opti = Description()
    # Time optimal control
    if not time_as_parameter:
        T_final = opti.sym("T_final", 1, lb=5, ub=1e2, x0=20)
        # Cost: time only
        J = T_final
        eps = 0.01
    else:
        T_final = opti.add_parameters("T_final", 1, values=15)
        J = 0
        eps = 0

    h = T_final/(N_stages*N_control_intervals*N_finite_elements)

    Xk = opti.sym("Xk", n_x, lb=X0, ub=X0, x0=X0)
    Yk = opti.sym_bool("Yk", n_y)
    for i in range(n_y):
        opti.eq(Yk[i], Y0[i])

    for _ in range(N_stages):
        Ykp = Yk
        Yk = opti.sym_bool("Yk", n_y)
        opti.add_g(1-eps, ca.sum1(Yk), 1+eps)  # SOS1
        # Transition condition
        LknUp = opti.sym_bool("LknUp", n_y-1)
        LknDown = opti.sym_bool("LknDown", n_y-1)
        # Transition
        LkUp = opti.sym_bool("LkUp", n_y-1)
        LkDown = opti.sym_bool("LkDown", n_y-1)

        psi = psi_fun(Xk)
        for i in range(n_y-1):
            # Trigger
            # If psi > psi_on -> psi - psi_on >= 0 -> LknUp = 1
            opti.higher_if_on(LknUp[i], psi - psi_on[i])
            opti.higher_if_on(LknDown[i], psi_off[i] - psi)
            # Only if trigger is ok, go up
            opti.leq(LkUp[i], LknUp[i])
            opti.leq(LkDown[i], LknDown[i])
            # # Only go up if active - already constraint using the
            # # Yk[i] = Ykprev + ...
            opti.leq(LkUp[i], Ykp[i])
            opti.leq(LkDown[i], Ykp[i+1])
            # Force going up if trigger = 1 and in right state!
            opti.leq(LknUp[i] + Ykp[i] - 1, LkUp[i])
            opti.leq(LknDown[i] + Ykp[i + 1] - 1, LkDown[i])
            # # Also force jump of two:
            # if i > 0:
            #     opti.leq(LknUp[i] + LknUp[i-1] + Ykp[i-1] - 2, LkUp[i])
            # if i < n_y - 2:
            #     opti.leq(LknDown[i] + LknDown[i+1] + Ykp[i+2] - 2, LkDown[i])

        for i in range(n_y):
            prev = Ykp[i]
            if i > 0:
                prev += LkUp[i-1] - LkDown[i-1]
            if i < n_y - 1:
                prev += LkDown[i] - LkUp[i]

            opti.eq(prev, Yk[i])

        for i in range(n_y):
            # # Add bounds to avoid late switch! (tolerance of 1)
            eps_bnd = 1
            if i > 0:
                opti.leq(Yk[i] - 1, (psi + eps_bnd - psi_on[i-1]) / v_max)

        for _ in range(N_control_intervals):
            Uk = opti.sym("U", n_u, lb=lbu, ub=ubu, x0=U0)

            for j in range(N_finite_elements):
                if not use_collocation:
                    # NEWTON FWD
                    Xkp = Xk
                    Xk = opti.sym("Xk", n_x, lb=lbx, ub=ubx, x0=X0)
                    if use_big_M:
                        for ii in range(n_y):
                            eq = Xk - (Xkp + h * F_dyn[ii](Xk, Uk))
                            for iii in range(n_x):
                                opti.equal_if_on(Yk[ii], eq[iii])
                    else:
                        eq = 0
                        for ii in range(n_y):
                            eq += Yk[ii] * F_dyn[ii](Xk, Uk)

                        opti.eq(Xk - Xkp, h * eq)
                else:
                    print("Collocation")
                    Xk_end = 0
                    X_fe = [
                        opti.sym("Xc", n_x, lb=lbx, ub=ubx, x0=X0)
                        for _ in range(n_s)
                    ]
                    for j in range(n_s):
                        xj = C_irk[0, j + 1] * Xk
                        for r in range(n_s):
                            xj += C_irk[r + 1, j + 1] * X_fe[r]

                        Xk_end += D_irk[j + 1] * X_fe[j]
                        if use_big_M:
                            print("with big M")
                            for iy in range(n_y):
                                eq = h * F_dyn[iy](X_fe[j], Uk) - xj
                                for iii in range(n_x):
                                    opti.equal_if_on(Yk[iy], eq[iii])
                        else:
                            eq = 0
                            for iy in range(n_y):
                                eq += Yk[iy] * F_dyn[iy](xj, Uk)

                            opti.add_g(-1e-7, h * eq - xj, 1e-7)

                    # J = J + L*B_irk*h;
                    Xk = opti.sym("Xk", n_x, lb=lbx, ub=ubx, x0=X0)
                    opti.eq(Xk, Xk_end)
        Ykp = Yk

    # Terminal constraints:
    slack1 = opti.sym('slack', 1, lb=0, ub=1)
    slack2 = opti.sym('slack', 1, lb=0, ub=1)
    opti.leq(Xk[0] - q_goal, slack1)
    opti.leq(q_goal - Xk[0], slack1)
    opti.leq(Xk[1] - v_goal, slack2)
    opti.leq(v_goal - Xk[1], slack2)
    opti.eq(Yk[0], 1)
    # J: value function = time
    opti.f = J + slack1 + slack2
    return opti


def run_bonmin(problem1=True):
    """Run bonmin."""
    opti = create_problem(use_big_M=True, more_stages=False, problem1=problem1)
    opti.set_solver(ca.nlpsol, 'bonmin', is_discrete=True,
                    options={"bonmin.time_limit": 7200})
    tic()
    opti.solve()
    opti.runtime = toc()
    return opti, opti.get("T_final")


def run_ipopt():
    """Run ipopt."""
    opti = create_problem()
    opti.set_solver(ca.nlpsol, 'ipopt', is_discrete=False)
    tic()
    opti.solve()
    opti.runtime = toc()
    return opti, opti.get("T_final")


def run_gurobi(problem1=True):
    """Run gurobi."""
    opti = create_problem(time_as_parameter=True, use_big_M=True, more_stages=True, problem1=problem1)
    opti.set_solver(
        ca.qpsol, "gurobi", is_discrete=True,
        options={
            "error_on_fail": False,
            # "gurobi": {
            #     "Threads": 1,
            # }
        }
    )
    T_max = 40
    T_min = 1
    tolerance = 1e-5

    lb_k = [T_min]
    ub_k = [T_max]
    solution = None
    T_opt = None
    itr = 0
    tic()
    while ub_k[-1] - lb_k[-1] > tolerance:
        itr += 1
        T_new = (ub_k[-1] + lb_k[-1]) / 2
        opti.set_parameters("T_final", T_new)
        opti.solve()
        if opti.is_solved():
            print(f"SUCCES {T_new=}")
            # Success
            ub_k.append(T_new)
            T_opt = T_new
            solution = opti.solution
        else:
            print(f"INF {T_new=}")
            lb_k.append(T_new)

    runtime = toc()
    print(f"TOLERANCE {ub_k[-1] - lb_k[-1]} - {itr=}")
    opti.solution = solution
    opti.runtime = runtime
    return opti, T_opt


if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: gurobi/bonmin <1/2> <outputfile>")
        exit(1)

    gurobi = "gur" in argv[1]
    problem1 = "1" in argv[2]
    outputfile = argv[3]
    print(f"{gurobi=} {problem1=} {outputfile=}")

    if gurobi:
        opti, T_final = run_gurobi(problem1=problem1)
    else:
        opti, T_final = run_bonmin(problem1=problem1)
    print(T_final)
    Xk = np.array(opti.get("Xk"))
    # plt.plot(Xk)
    # plt.show()
    print(opti.get("Yk"))
    data = {
        key: opti.get(key)
        for key in opti.indices.keys()
    }
    data["T_final"] = T_final
    data["runtime"] = opti.runtime
    with open(outputfile, "wb") as f:
        pickle.dump(data, f)
