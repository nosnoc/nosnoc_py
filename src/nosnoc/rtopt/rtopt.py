from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
from time import monotonic

class WarmstartType(Enum):
    NONE              = auto() #: Don't warmstart the nonlinear problem.
    WARMSTART_PRIMALS = auto() #: Warmstart only the primal values with previous result.
    WARMSTART_ALL     = auto() #: Warmstart primals and (generic constraint) duals with previous result.
    SHIFT             = auto() #: Warmstart by shifting the previous result

@dataclass
class RealtimeOptimizationStats():
    update_time: List[float] = field(default_factory=list) #: Total time spent in the "feedback" (using the mpc terms) phase.
    prepare_time: List[float]  = field(default_factory=list) #: Total time spent in the "preparation" phase.\


class RealtimeOptimizationAlgorithm(ABC):
    """
    Base class for real-time optimization algorithms including the MPC and MHE.
    """
    def __init__(self, model, ocp_opts, rt_opts):
        self.model = model
        self.ocp_opts = ocp_opts
        self.rt_opts = rt_opts

    def update(self, **kwargs):
        """
        The "correction" phase of an real-time optimization algorithm where the next output is calculated.
        """
        start = monotonic()
        ret = self._update(**kwargs)
        self.stats.update_time.append(monotonic() - start)
        return ret

    def prepare(self, **kwargs):
        """
        The "preparation" phase of an real-time optimization algorithm where numerical data is calculated in preparation for the next update.
        """

        start = monotonic()
        ret = self._prepare(**kwargs)
        self.stats.prepare_time.append(monotonic() - start)
        return ret

    @abstractmethod
    def _update(self, **kwargs):
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
