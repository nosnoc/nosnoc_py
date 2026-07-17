from abc import ABC, abstractmethod
from typing import override

import numpy as np

class MpccRelaxation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def relax(self, w1, w2, sigma):
        """
        Returns (relaxation expression, lb, ub), (w1 bounds expression, lb, ub), (w2 bounds expression, lb, ub)
        """
        pass

class ScholtesRelaxation(MpccRelaxation):
    def __init__(self, inequality=True):
        self.inequality=True
    @override
    def relax(self, w1, w2, sigma):
        e = np.ones(w1.size(1))
        if self.inequality:
            return (w1*w2 - sigma, -np.inf*e, 0.0*e), (w1, 0.0*e, np.inf*e), (w2, 0.0, np.inf*e)
        else:
            return (w1*w2 - sigma, 0.0*e, 0.0*e), (w1, 0.0*e, np.inf*e), (w2, 0.0*e, np.inf*e)
