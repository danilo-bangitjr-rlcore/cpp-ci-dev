import logging

import numpy as np
from numba import njit
from numpy import ndarray

logger = logging.getLogger(__name__)


class ExpMovingAvg:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._mu: float = np.nan

    def __call__(self) -> float:
        return self._mu

    def feed(self, x: ndarray) -> None:
        self._mu = _stream_trace_update(self._mu, x, self.alpha)


@njit
def _stream_trace_update(initial_trace: float, new_vals: ndarray, alpha: float) -> float:
    trace = initial_trace
    if np.isnan(trace):
        first_val_idx = _find_first_val_idx(new_vals)
        # return if there are no non-nan vals
        if first_val_idx >= len(new_vals):
            return np.nan

        # else, initialize trace
        else:
            assert isinstance(first_val_idx, int)
            trace = new_vals[first_val_idx]
            new_vals = new_vals[first_val_idx + 1 :]  # start streaming from next val

    # prune nans
    new_vals = new_vals[~np.isnan(new_vals)]

    for val in new_vals:
        trace = (1 - alpha) * val + alpha * trace

    return trace

@njit
def _find_first_val_idx(x: ndarray) -> int:
    """
    The convention here (for numba typing purposes) is that
    returning an index >= len(x) means that x is full of nans
    """
    i = 0
    x_i = x[i]
    while np.isnan(x_i):
        i += 1
        if i < len(x):
            x_i = x[i]
        else:
            break

    return i

class ExpMovingVar:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._ema = ExpMovingAvg(alpha)
        self._var: float = np.nan

    def __call__(self) -> float:
        assert self._var is not None
        return self._var

    def feed(self, x: ndarray) -> None:
        self._ema.feed(x)
        square_residuals = (self._ema() - x) ** 2
        self._var = _stream_trace_update(self._var, square_residuals, self.alpha)
