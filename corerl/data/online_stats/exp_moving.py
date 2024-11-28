import logging

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)


class ExpMovingAvg:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mu: float | None = None

    def __call__(self) -> float:
        assert self.mu is not None
        return self.mu

    def feed(self, x: float | ndarray) -> None:
        batch_avg = x.mean() if isinstance(x, ndarray) else x
        if self.mu is None:
            self.mu = batch_avg
        else:
            self.mu = (1 - self.alpha) * batch_avg + self.alpha * self.mu


class ExpMovingVar:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.ema = ExpMovingAvg(alpha)
        self.var: float | None = None

    def __call__(self) -> float:
        assert self.var is not None
        return self.var

    def _get_batch_var(self, x: float | ndarray):
        """
        gets variance with respect to exponential moving average (rather than wrt the batch mean)
        """
        mu = self.ema()
        residuals = mu - x
        if isinstance(x, ndarray):
            batch_var = np.square(residuals).mean()
        else:
            assert isinstance(residuals, float)
            batch_var = residuals**2

        return batch_var

    def feed(self, x: float | ndarray) -> None:
        self.ema.feed(x)
        batch_var = self._get_batch_var(x)
        if self.var is None:
            self.var = batch_var
        else:
            self.var = (1 - self.alpha) * batch_var + self.alpha * self.var
