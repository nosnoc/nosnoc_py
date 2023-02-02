from dataclasses import dataclass


@dataclass
class NosnocDims:
    """
    detected automatically
    """
    n_x: int
    n_u: int
    n_z: int
    n_sys: int
    n_p_time_var: int
    n_p_glob: int
    n_c_sys: list
    n_f_sys: list
