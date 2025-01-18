from collections import defaultdict
from typing import Literal

import numpy as np
from numba import njit

from corerl.configs.config import config
from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class NormalizerConfig(BaseTransformConfig):
    name: Literal['normalize'] = 'normalize'
    min: float | None = None
    max: float | None = None
    from_data: bool = True


class Normalizer:
    def __init__(self, cfg: NormalizerConfig):
        self._cfg = cfg
        self._mins = defaultdict(lambda: cfg.min)
        self._maxs = defaultdict(lambda: cfg.max)

    def reset(self):
        self._mins.clear()
        self._maxs.clear()

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            if np.isnan(x).all():
                self._assign_to_df(carry, col, x)
                continue

            if self._cfg.from_data:
                self._update_bounds(col, x)

            mi = self._mins[col]
            ma = self._maxs[col]
            assert mi is not None and ma is not None

            den = ma - mi
            assert den > 1e-12, f'Not enough variability in the data to normalize: {carry.tag}'

            new_x = _norm(x, mi, ma)
            self._assign_to_df(carry, col, new_x)

        return carry, None


    def _assign_to_df(self, carry: TransformCarry, col: str, x: np.ndarray):
        new_name = f'{col}_norm'
        carry.transform_data.drop(col, axis=1, inplace=True)
        carry.transform_data[new_name] = x


    def _update_bounds(self, col: str, x: np.ndarray):
        mi = self._mins[col]
        mi = mi if mi is not None else np.inf
        self._mins[col] = min(mi, np.nanmin(x))

        ma = self._maxs[col]
        ma = ma if ma is not None else -np.inf
        self._maxs[col] = max(ma, np.nanmax(x))


transform_group.dispatcher(Normalizer)

@njit
def _norm(x: np.ndarray, mi: float, ma: float):
    n = len(x)
    out = np.zeros(n, dtype=np.float_)

    for i in range(n):
        if np.isnan(x[i]):
            out[i] = np.nan
            continue

        out[i] = (x[i] - mi) / (ma - mi)

    return out
