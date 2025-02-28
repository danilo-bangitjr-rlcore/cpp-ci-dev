import numba
import numpy as np


@numba.njit
def put_in_range(
    x: np.ndarray | float,
    old_range: tuple[float, float] | np.ndarray,
    new_range: tuple[float, float] | np.ndarray,
):
    """
    Take a float value x in old_range and map it to a value in new_range.
    E.g. if x is 0.5 in [0, 1] and new_range is [-1, 1], then return 0.
    """
    old_d = (old_range[1] - old_range[0])
    new_d = (new_range[1] - new_range[0])
    return (((x - old_range[0]) * new_d) / old_d) + new_range[0]
