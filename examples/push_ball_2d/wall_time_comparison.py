"""Wall time per homotopy step of the 2d push ball OCP, Pds vs. Cls model.

Run from the repository root:  python -m examples.cls_minimal_example.push_ball_2d_wall_time
"""
import matplotlib.pyplot as plt

from examples.push_ball_2d import push_ball_pds as pds
from examples.push_ball_2d import push_ball_cls as cls

for name, example in [("Pds", pds), ("Cls", cls)]:
    stats = example.solve_ocp().dtp.solver.stats
    plt.plot(stats["t_wall"], "o-", label=f"{name} (total {stats['wall_time_total']:.2f} s)")

plt.xlabel("homotopy step")
plt.ylabel("wall time [s]")
plt.legend()
plt.grid()
plt.show()
