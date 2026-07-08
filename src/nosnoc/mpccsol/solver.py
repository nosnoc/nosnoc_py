from abc import ABC, abstractmethod
from typing import Optional, List

import casadi as ca
import numpy as np

import nosnoc

def mpccsol(plugin:str, mpcc, opts):
    if plugin == "reg_homotopy":
        return nosnoc.mpccsol.plugins.reg_homotopy.RegHomotopySolver(mpcc, opts)
