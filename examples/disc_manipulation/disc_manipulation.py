import numpy as np
from casadi import SX, horzcat
import matplotlib.pyplot as plt

import nosnoc



def get_default_options(**kwargs):
    N_stg = 15
    N_finite_elements = 2
    n_s = 2
    T = 2
    
    default_args = {
        "N_stages": N_stg,
        "N_finite_elements": N_finite_elements,
        "N_stages_RK": n_s,
        "T": T,
        "use_fesd": True,
        "cross_comp_mode": nosnoc.CrossComplementarityMode.FE_FE,
        }
    
    merged = dict(list(default_args.items())+ list(kwargs.items()))
    # switch
    opts = nosnoc.Options(
        **merged
    )
    return opts