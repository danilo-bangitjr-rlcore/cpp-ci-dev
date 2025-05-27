from typing import Literal

import numpy as np
from numba import njit

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class LessThanConfig(BaseTransformConfig):
    name: Literal['less_than'] = 'less_than'
    threshold: float = 0.0
    equal: bool = False # if true, <=


class LessThan:
    def __init__(self, cfg: LessThanConfig):
        self._cfg = cfg
        self._threshold = cfg.threshold
        self._equal = cfg.equal

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            new_x = _less_than(x, self._threshold, self._equal)

            symbol = "<" if not self._equal else "<="
            new_name = f'{col}{symbol}{self._threshold}'
            carry.transform_data.drop(col, axis=1, inplace=True)
            carry.transform_data[new_name] = new_x

        return carry, None

    def reset(self) -> None:
        pass

transform_group.dispatcher(LessThan)

@njit
def _less_than(x: np.ndarray, threshold: float, equal: bool):
    n = len(x)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isnan(x[i]):
            out[i] = np.nan
            continue

        if equal:
            out[i] = x[i] <= threshold
        else:
            out[i] = x[i] < threshold

    return out
