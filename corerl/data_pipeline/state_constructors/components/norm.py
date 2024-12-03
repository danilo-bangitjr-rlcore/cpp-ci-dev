import numpy as np
from numba import njit
from collections import defaultdict
from dataclasses import dataclass

from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, sc_group
from corerl.data_pipeline.state_constructors.interface import TransformCarry


@dataclass
class NormalizerConfig(BaseTransformConfig):
    name: str = 'normalize'
    min: float | None = None
    max: float | None = None
    from_data: bool = True


class Normalizer:
    def __init__(self, cfg: NormalizerConfig):
        self._cfg = cfg
        self._mins = defaultdict(lambda: cfg.min)
        self._maxs = defaultdict(lambda: cfg.max)

    def __call__(self, carry: TransformCarry, ts: object | None):
        cols = set(carry.agent_state.columns)
        for col in cols:
            x = carry.agent_state[col].to_numpy()

            if self._cfg.from_data:
                mi = self._mins[col]
                mi = mi if mi is not None else np.inf
                self._mins[col] = min(mi, np.nanmin(x))

                ma = self._maxs[col]
                ma = ma if ma is not None else -np.inf
                self._maxs[col] = max(ma, np.nanmax(x))

            mi = self._mins[col]
            ma = self._maxs[col]
            assert mi is not None and ma is not None

            den = ma - mi
            assert den > 1e-12, 'Not enough variability in the data to normalize'

            new_x = _norm(x, mi, ma)

            new_name = f'{col}_norm'
            carry.agent_state.drop(col, axis=1, inplace=True)
            carry.agent_state[new_name] = new_x

        return carry, None

sc_group.dispatcher(Normalizer)

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
