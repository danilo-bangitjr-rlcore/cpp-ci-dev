import logging
import numpy as np

logger = logging.getLogger(__name__)


class ExpMovingAvg:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._mu: float | None = None

    def feed(self, x: np.ndarray) -> None:
        if self._mu is None:
            first_valid_idx = np.where(~np.isnan(x))[0]
            if len(first_valid_idx) > 0:
                self._mu = float(x[first_valid_idx[0]])
            return

        nan_counts = np.zeros_like(x)
        nan_count = 0

        for i in range(len(x)):
            if np.isnan(x[i]):
                nan_count += 1
            else:
                nan_counts[i] = nan_count
                nan_count = 0

        for i in range(len(x)):
            if not np.isnan(x[i]):
                effective_alpha = self.alpha ** (nan_counts[i] + 1)
                self._mu = float((1 - effective_alpha) * x[i] + effective_alpha * self._mu)

    def __call__(self) -> float:
        if self._mu is None:
            return 0.0
        return self._mu


class ExpMovingVar:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._var: float | None = None
        self._ema: ExpMovingAvg = ExpMovingAvg(alpha)
        self._prev_mean: float = 0.0

    def feed(self, x: np.ndarray) -> None:
        if self._var is None:
            first_valid_idx = np.where(~np.isnan(x))[0]
            if len(first_valid_idx) > 0:
                self._var = 1e-6
                self._ema.feed(x)
                self._prev_mean = self._ema()
            return

        nan_counts = np.zeros_like(x)
        nan_count = 0

        for i in range(len(x)):
            if np.isnan(x[i]):
                nan_count += 1
            else:
                nan_counts[i] = nan_count
                nan_count = 0

        self._prev_mean = self._ema()
        self._ema.feed(x)
        for i in range(len(x)):
            if not np.isnan(x[i]):
                effective_alpha = self.alpha ** (nan_counts[i] + 1)
                delta = x[i] - self._prev_mean
                self._var = float((1 - effective_alpha) * (delta * delta) + effective_alpha * self._var)

    def __call__(self) -> float:
        if self._var is None:
            return 1e-6
        return max(1e-6, self._var)
