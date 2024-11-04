from dataclasses import dataclass
import numpy as np
import logging

from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod

from omegaconf import MISSING

from corerl.utils.types import Ring

log = logging.getLogger(__name__)


R = TypeVar('R', bound=Ring)


@dataclass
class NormalizerConfig:
    action_normalizer: Any = MISSING
    obs_normalizer: Any = MISSING
    reward_normalizer: Any = MISSING

# ----------
# -- Base --
# ----------
class BaseNormalizer(ABC, Generic[R]):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: R) -> R:
        raise NotImplementedError


class InvertibleNormalizer(BaseNormalizer[R]):
    @abstractmethod
    def denormalize(self, x: R) -> R:
        raise NotImplementedError


# --------------
# -- Identity --
# --------------
class Identity(InvertibleNormalizer):
    def __init__(self):
        return

    def __call__(self, x: R) -> R:
        return x

    def denormalize(self, x: R) -> R:
        return x


# -----------
# -- Scale --
# -----------
class Scale(InvertibleNormalizer):
    def __init__(self, scale: R, bias: R):
        self.scale = scale
        self.bias = bias

    def __call__(self, x: R) -> R:
        return (x - self.bias) / self.scale

    def denormalize(self, x: R) -> R:
        return x * self.scale + self.bias


# ----------
# -- Clip --
# ----------
class Clip(BaseNormalizer):
    def __init__(self, min_: R, max_: R):
        self.min: Any = min_
        self.max: Any = max_

    def __call__(self, x: Any) -> Any:
        return np.clip(x, self.min, self.max)


# ------------
# -- OneHot --
# ------------
class OneHot(InvertibleNormalizer):
    def __init__(self, total_count: int, start_from: int):
        self.total_count = total_count
        self.start = start_from

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 and x.shape[1] == 1  # shape = batch_size * 1
        oneh = np.zeros((x.shape[0], self.total_count))
        oneh[np.arange(x.shape[0]), (x - self.start).astype(int).squeeze()] = 1
        return oneh

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        idx = np.where(x == 1)[1]
        idx = np.expand_dims(idx, axis=1)
        return idx


# ------------
# -- MaxMin --
# ------------
class MaxMin(InvertibleNormalizer):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max
        self.scale = self.max - self.min
        self.bias = self.min

    def __call__(self, x: R) -> R:
        return (x - self.bias) / self.scale

    def denormalize(self, x: R) -> R:
        return x * self.scale + self.bias


# ----------------
# -- AvgNanNorm --
# ----------------
class AvgNanNorm(InvertibleNormalizer):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max
        self.scale = self.max - self.min
        self.bias = self.min

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.handle_nan(x)
        # x = self.average(x)
        return self.normalize(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.bias) / self.scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.bias

    def handle_nan(self, x: np.ndarray) -> np.ndarray:
        # Adapted from https://stackoverflow.com/a/62039015
        fill = np.nanmean(x, axis=0)  # try to fill with the mean in each column
        # If an entire column is nan, fill with zeros
        fill_mask = np.isnan(fill)
        fill[fill_mask] = np.zeros_like(fill)[fill_mask]

        mask = np.isnan(x[0])
        x[0][mask] = fill[mask]
        for i in range(1, len(x)):
            mask = np.isnan(x[i])
            x[i][mask] = x[i - 1][mask]
        return x

    def average(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 2:
            return np.mean(x, axis=0)
        elif len(x.shape) == 1:
            return x
        else:
            raise ValueError("Invalid shape for observation")
