from abc import ABC, abstractmethod
from typing import Optional, List

import casadi as ca
import numpy as np

from .reg_homotopy import RegHomotopySolver

def mpccsol(plugin:str, mpcc, opts):
    if plugin == "reg_homotopy":
        return RegHomotopySolver(mpcc, opts)

class MpccsolPlugin(ABC):

    def __init__(self, mpcc, opts):
        self.mpcc = mpcc
        self.opts = opts
