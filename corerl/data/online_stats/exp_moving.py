from numpy import ndarray
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ExpMovingBatchAvg:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mu: float | None = None

    def __call__(self) -> float:
        assert self.mu is not None
        return self.mu

    def feed(self, x: ndarray) -> None:
        batch_avg = x.mean()
        if self.mu is None:
            self.mu = batch_avg
        else:
            self.mu = (1 - self.alpha) * batch_avg + self.alpha * self.mu

class ExpMovingBatchVar:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.ema = ExpMovingBatchAvg(alpha)
        self.var: float | None = None

    def __call__(self) -> float:
        assert self.var is not None
        return self.var

    def _get_batch_var(self, x: ndarray):
        """
        gets variance with respect to exponential moving average (rather than wrt the batch mean)
        """
        mu = self.ema()
        residuals = mu - x
        return np.square(residuals).mean()


    def feed(self, x: ndarray) -> None:
        self.ema.feed(x)
        batch_var = self._get_batch_var(x)
        if self.var is None:
            self.var = batch_var
        else:
            self.var = (1 - self.alpha) * batch_var + self.alpha * self.var
