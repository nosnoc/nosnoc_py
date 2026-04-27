from .base import Base

class Pss(Base):

    def __init__(self,
                 F: List[ca.SX],
                 S: List[np.ndarray],
                 c,
                 g_indicator,
                 f_0
                 ):
