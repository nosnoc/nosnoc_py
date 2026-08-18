from abc import ABC, abstractmethod

import casadi as ca
import numpy as np

class MpccsolPlugin(ABC):
    def __init__(self, mpcc, opts):
        self.mpcc = mpcc
        self.opts = opts
        self.stats = {}
        self._build_solver()

    @abstractmethod
    def _build_solver(self):
        pass

    @abstractmethod
    def _solve(self,
               x0:     np.ndarray,
               y0:     np.ndarray, # Note: y0 is unused and is the initial complementarity active set
               lbx:    np.ndarray,
               ubx:    np.ndarray,
               lbg:    np.ndarray,
               ubg:    np.ndarray,
               p:      np.ndarray,
               lam_g0: np.ndarray,
               lam_x0: np.ndarray,
               ):
        pass

    def __call__(self, x0=None, y0=None, lbx=None, ubx=None, lbg=None, ubg=None, p=None, lam_g0=None, lam_x0=None):
        if x0 is None:
            x0 = np.zeros(len(self.mpcc.w))
        if y0 is None:
            y0 = np.zeros(len(self.mpcc.G), dtype=bool)
        if lbx is None:
            lbx = -np.inf*np.ones(len(self.mpcc.w))
        if ubx is None:
            ubx = np.inf*np.ones(len(self.mpcc.w))
        if lam_x0 is None:
            lam_x0 = np.zeros(len(self.mpcc.w))
        if lbg is None:
            lbg = np.zeros(len(self.mpcc.g))
        if ubg is None:
            ubg = np.zeros(len(self.mpcc.g))
        if lam_g0 is None:
            lam_g0 = np.zeros(len(self.mpcc.g))
        if p is None:
            p = np.zeros(len(self.mpcc.p))

        # TODO(@anton) check dimensions
        # TODO(@anton) this is wasteful of memory, pass through Nones instead
        return self._solve(
            x0,
            y0, # Note y0 is unused and is the initial complementarity active set
            lbx,
            ubx,
            lbg,
            ubg,
            p,
            lam_g0,
            lam_x0,
        )
