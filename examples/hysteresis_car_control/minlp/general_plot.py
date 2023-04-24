"""General plotting."""

from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import pickle


def interp0(x, xp, yp):
    """Create zeroth order hold interpolation w/ same base signature as numpy.interp."""
    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while k < len(xp) and k < len(yp) and x0 > xp[k]:
            k += 1
        return yp[k-1]

    if isinstance(x, float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')


def plot(x_list, y_list, T_final, u):
    """Plot."""
    q = x_list[:, 0]
    v = x_list[:, 1]
    print("error")
    print(np.sqrt(300-q[-1])**2 + v[-1]**2)
    # L = x_list[:,2]
    aux = np.zeros((y_list.shape[0],))
    for i in range(y_list.shape[1]):
        aux += i * y_list[:, i]

    N = x_list.shape[0]
    t = [T_final / (N - 1) * i for i in range(N)]
    N_fe = int((len(t) - 1) / (len(aux) - 1))
    t_grid_aux = [t[i] for i in range(0, len(t), N_fe)][:-1]
    aux = aux[1:]

    plt.figure()
    plt.plot(t, q)
    v_aux = [max((q[i+1] - q[i]) / max(1e-9, t[i+1] - t[i]), 0)
             for i in range(len(t)-1)]

    plt.figure()
    plt.plot(t[:-1], v_aux, label="V recalc")
    plt.plot(t, v, label="V actual")
    plt.legend()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, v)
    plt.xlabel("$t$")
    plt.ylabel("$v(t)$")
    plt.subplot(2, 2, 2)
    plt.plot(t, interp0(t, t_grid_aux, u))
    plt.xlabel("$t$")
    plt.ylabel("$u(t)$")
    plt.subplot(2, 2, 3)
    plt.scatter(t_grid_aux, aux)
    plt.plot(t_grid_aux, aux)
    plt.xlabel("$t$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.subplot(2, 2, 4)
    plt.scatter([v[i] for i in range(0, len(t)-1, N_fe)], aux)
    plt.plot([v[i] for i in range(0, len(t)-1, N_fe)], aux)
    plt.xlabel("$v$")
    plt.ylabel("$w_1(t) + w_2(t)$")
    plt.show()


if len(argv) <= 1:
    file = "data_minlp.pickle"
else:
    file = argv[1]

with open(file, "rb") as f:
    results = pickle.load(f)

try:
    print(f"{results['runtime']}")
except:
    pass
# breakpoint()
# results['slack']
plt.subplot(6, 1, 1)
plt.plot(np.array(results["Xk"])[:, 1])
for i, name in enumerate(["LknUp", "LkUp", "LknDown", "LkDown", "Yk"]):
    plt.subplot(6, 1, i+2)
    var = np.array(results[name])
    plt.plot(
        var, label=[f"{name}{j}" for j in range(var.shape[1])]
    )
    plt.legend()

plt.show()

print(results['T_final'])
plot(
    np.array(results["Xk"]),
    np.array(results["Yk"]),
    results['T_final'],
    np.array(results["U"]),
)
