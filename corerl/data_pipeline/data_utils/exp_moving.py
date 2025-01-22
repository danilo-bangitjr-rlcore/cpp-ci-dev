import logging

import numpy as np

logger = logging.getLogger(__name__)


class ExpMovingAvg:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mu: float = np.nan
        self._nan_count: int = 0

    def feed_single(self, x: float) -> None:
        if np.isnan(x):
            self._nan_count += 1
            return

        if np.isnan(self.mu):
            self.mu = x
        else:
            effective_alpha = self.alpha ** (self._nan_count + 1)
            self.mu = (1 - effective_alpha) * x + effective_alpha * self.mu

        self._nan_count = 0

    def feed(self, x: np.ndarray) -> None:
        for x_i in x:
            self.feed_single(x_i)



class ExpMovingVar:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.var: float = np.nan
        self._ema: ExpMovingAvg = ExpMovingAvg(alpha)
        self._nan_count: int = 0

    def feed_single(self, x: float) -> None:
        self._ema.feed_single(x)

        if np.isnan(x):
            self._nan_count += 1
            return

        mu = self._ema.mu
        delta = x - mu
        if np.isnan(self.var):
            self.var = delta ** 2
        else:
            effective_alpha = self.alpha ** (self._nan_count + 1)
            self.var = (1 - effective_alpha) * delta ** 2 + effective_alpha * self.var

        self._nan_count = 0

    def feed(self, x: np.ndarray) -> None:
        for x_i in x:
            self.feed_single(x_i)

