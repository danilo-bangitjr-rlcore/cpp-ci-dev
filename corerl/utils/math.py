from typing import Any

import numba
import numpy as np

ArrayOrFloat = np.ndarray | float | np.floating

@numba.njit
def put_in_range(
    x: ArrayOrFloat,
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


@numba.njit
def exp_moving_avg[T: ArrayOrFloat](
    decay: float,
    mean: T | None,
    x: ArrayOrFloat,
) -> T:
    """
    Computes an exponential moving average:
        mean = decay * mean + (1 - decay) * x
    """
    if mean is None:
        # using type erasure to guarantee return type
        out: Any = x
        return out

    out: Any = decay * mean + (1 - decay) * x
    return out
