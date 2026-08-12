from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from time import monotonic

@dataclass
class RealtimeOptimizationStats():
    optimize_time: List[float] = field(default_factory=list)
    prepare_time: List[float] = field(default_factory=list)


class RealtimeOptimizationAlgortihm(ABC):
    """
    Base class for real-time optimization algorithms including the MPC and MHE.
    """
    def __init__(self, model, ocp_opts, rt_opts):
        self.model = model
        self.ocp_opts = ocp_opts
        self.rt_opts = rt_opts

    def optimize(self, **kwargs):
        start = monotonic()
        ret = self._optimize(**kwargs)
        self.stats.optimize_time.push(monotonic() - start)
        return ret

    def prepare(self, **kwargs):
        start = monotonic()
        ret = self._prepare(**kwargs)
        self.stats.prepare_time.push(monotonic() - start)
        return ret

    @abstractmethod
    def _optimize(self, **kwargs):
        """
        Take single optimization step for the current time step parameters.
        """
        raise NotImplementedError("")

    @abstractmethod
    def _prepare(self, **kwargs):
        """
        Prepare for next time step's parameters using the given predicted parameters.
        """
        raise NotImplementedError("")
