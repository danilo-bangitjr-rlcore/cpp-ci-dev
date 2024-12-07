import numpy as np
from numba import njit
from dataclasses import dataclass

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@dataclass
class GreaterThanConfig(BaseTransformConfig):
    name: str = 'greater_than'
    threshold: float = 0.0


class GreaterThan:
    def __init__(self, cfg: GreaterThanConfig):
        self._cfg = cfg
        self._threshold = cfg.threshold

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            new_x = _greater_than(x, self._threshold)

            new_name = f'{col}_lessthan_{self._threshold}'
            carry.transform_data.drop(col, axis=1, inplace=True)
            carry.transform_data[new_name] = new_x

        return carry, None

transform_group.dispatcher(GreaterThan)

@njit
def _greater_than(x: np.ndarray, threshold: float):
    n = len(x)
    out = np.zeros(n, dtype=np.float_)

    for i in range(n):
        if np.isnan(x[i]):
            out[i] = np.nan
            continue

        out[i] = x[i] > threshold

    return out
