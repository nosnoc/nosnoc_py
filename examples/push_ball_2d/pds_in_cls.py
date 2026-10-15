import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import push_ball_pds, push_ball_cls



u_pds = push_ball_pds.solve_ocp().get("u")





