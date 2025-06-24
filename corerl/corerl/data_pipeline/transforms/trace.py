from dataclasses import dataclass
from typing import Literal

import numpy as np
from lib_config.config import config, list_
from lib_utils.maybe import Maybe
from numba import njit

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry


@config()
class TraceConfig(BaseTransformConfig):
    name: Literal['multi_trace'] = 'multi_trace'
    trace_values: list[float] = list_([0., 0.75, 0.9, 0.95])


@dataclass
class TraceTemporalState:
    mu: dict[str, np.ndarray] | None = None


class TraceConstructor:
    def __init__(self, cfg: TraceConfig):
        self._cfg = cfg
        self._decays = np.array(cfg.trace_values)

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, TraceTemporalState | None)
        mu = (ts and ts.mu) or {}

        cols = set(carry.transform_data.columns)
        for col in cols:
            x = carry.transform_data[col].to_numpy()
            mu_0 = Maybe(mu.get(col)).or_else(np.ones_like(self._decays, dtype=np.float64) * np.nan)
            assert x.ndim == 1, f"shape of column {col}: {x.shape}, transform_data: {carry.transform_data}"

            trace_vals, new_mu = compute_trace_with_nan(
                data=x,
                decays=self._decays,
                mu_0=mu_0,
            )

            mu[col] = new_mu
            carry.transform_data.drop(col, axis=1, inplace=True)

            for i, decay in enumerate(self._decays):
                new_name = f'{col}_trace-{decay}'
                carry.transform_data[new_name] = trace_vals[:, i]

        return carry, TraceTemporalState(mu)

    def reset(self) -> None:
        pass


transform_group.dispatcher(TraceConstructor)

@njit
def compute_trace_with_nan(
    data: np.ndarray,
    decays: np.ndarray,
    mu_0: np.ndarray,
):
    n_samples = len(data)
    n_traces = len(decays)
    out = np.zeros((n_samples, n_traces), dtype=np.float64)
    mu = mu_0

    for i in range(n_samples):
        x = data[i]

        # initialize trace if it has been reset to nan
        nanmask = np.isnan(mu)
        mu[nanmask] = np.ones(nanmask.sum()) * x

        # update trace
        mu = decays * mu + (1 - decays) * x
        out[i] = mu

    return out, mu
