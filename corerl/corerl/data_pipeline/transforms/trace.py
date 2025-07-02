from dataclasses import dataclass
from typing import Literal, NamedTuple

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
    missing_tol: float = 0.25 # proportion of the trace that can be "missing"


@dataclass
class TraceTemporalState:
    trace: dict[str, np.ndarray] | None = None

class TraceParams(NamedTuple):
    decays: np.ndarray

class TraceData(NamedTuple):
    trace: np.ndarray # size == (len(decays),)
    obs: np.ndarray # size == (n,)

class TraceConstructor:
    def __init__(self, cfg: TraceConfig):
        self._cfg = cfg

        decays = np.array(cfg.trace_values)

        self._trace_params = TraceParams(
            decays=decays,
        )

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, TraceTemporalState | None)
        trace = (ts and ts.trace) or {}

        cols = set(carry.transform_data.columns)
        for col in cols:
            obs = carry.transform_data[col].to_numpy()
            prev_trace = Maybe(trace.get(col)).or_else(np.ones_like(self._trace_params.decays, dtype=np.float64)*np.nan)
            assert obs.ndim == 1, f"shape of column {col}: {obs.shape}, transform_data: {carry.transform_data}"

            trace_data = TraceData(
                trace=prev_trace,
                obs=obs,
            )
            out, new_trace_data = compute_trace_with_nan(self._trace_params, trace_data)

            trace[col] = new_trace_data.trace
            carry.transform_data.drop(col, axis=1, inplace=True)

            for i, decay in enumerate(self._trace_params.decays):
                new_name = f'{col}_trace-{decay}'
                carry.transform_data[new_name] = out[:, i]

        return carry, TraceTemporalState(trace)

    def reset(self) -> None:
        pass


transform_group.dispatcher(TraceConstructor)

@njit
def compute_trace_with_nan(trace_params: TraceParams, trace_data: TraceData):
    n_samples = len(trace_data.obs)
    n_traces = len(trace_params.decays)
    out = np.zeros((n_samples, n_traces), dtype=np.float64)

    decays = trace_params.decays
    trace = trace_data.trace

    for i in range(n_samples):
        x = trace_data.obs[i]

        # initialize trace nans if this was the first obs
        trace_nanmask = np.isnan(trace)
        trace = np.where(trace_nanmask, x, trace)

        # update trace if obs is not nan, otherwise forward old trace
        obs_is_nan = np.isnan(x)
        trace = decays * trace + (1 - decays) * x if ~obs_is_nan else trace

        # output trace value
        out[i] = trace

    return out, TraceData(trace, trace_data.obs)
