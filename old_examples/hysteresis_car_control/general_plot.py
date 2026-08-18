from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from nosnoc.plot_utils import plot_colored_line_3d


def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while k < len(xp) and x0 > xp[k]:
            k += 1
        return yp[k-1]

    if isinstance(x,float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')


def filter_t(x, t):
    """Filter based on dt."""
    dt = [t2 - t1 for t1, t2 in zip(t, t[1:])]
    out = []
    for xi, dti in zip(x, dt):
        if dti > 1e-3:
            out.append(xi)
    out.append(x[-1])
    return out


def plot(x_list, t_grid, u_list, t_grid_u):
    """Plot."""
    q = [x[0] for x in x_list]
    v = [x[1] for x in x_list]
    w1 = [x[-2] for x in x_list]
    w2 = [x[-3] for x in x_list]
    aux = [x[-2] + x[-3] for x in x_list]
    t = [x[-1] for x in x_list]
    u = [u[0] for u in u_list]
    print("Error")
    print(np.sqrt(300-q[-1])**2 + v[-1]**2)
    try:
        s = [u[1] for u in u_list]
        print(s)
    except Exception:
        pass
    plt.figure()
    plt.plot(t, q)
    v_aux = [max(q[i+1] - q[i] / max(1e-9, t[i+1] - t[i]), 0) for i in range(len(t)-1)]

    ax = plot_colored_line_3d(v, w1, w2, t)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel("$v(t)$")
    ax.set_ylabel("$w_1(t)$")
    ax.set_zlabel("$w_2(t)$")

    plt.figure()
    plt.plot(t[:-1], v_aux)
    plt.plot(t, v)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(filter_t(t, t), filter_t(v, t))
    plt.xlabel("$t$")
    plt.ylabel("$v(t)$")
    plt.subplot(2, 2, 2)
    plt.plot(filter_t(t, t), filter_t(interp0(t_grid, t_grid_u, u), t))
    plt.xlabel("$t$")
    plt.ylabel("$u(t)$")
    plt.subplot(2, 2, 3)
    plt.plot(t, aux)
    plt.xlabel("$t$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.subplot(2, 2, 4)
    plt.plot(v, aux)
    plt.xlabel("$v$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.show()


if len(argv) <= 1:
    file = "data_3d.pickle"
else:
    file = argv[1]

with open(file, "rb") as f:
    results = pickle.load(f)

print(f"{results['v_global']=}")
plot(
    results["x_traj"], results["t_grid"],
    results["u_list"], results["t_grid_u"]
)
