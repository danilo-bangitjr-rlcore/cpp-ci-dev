from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from lib_config.config import config, list_
from lib_utils.maybe import Maybe
from numba import njit

from corerl.data_pipeline.transforms.base import BaseTransformConfig, transform_group
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.state import AppState


@config()
class TraceConfig(BaseTransformConfig):
    name: Literal['multi_trace'] = 'multi_trace'
    trace_values: list[float] = list_([0., 0.75, 0.9, 0.95])
    missing_tol: float = 0.25 # proportion of the trace that can be "missing"


@dataclass
class TraceTemporalState:
    trace: dict[str, np.ndarray] | None = None
    quality: dict[str, np.ndarray] | None = None

class TraceParams(NamedTuple):
    decays: np.ndarray
    missing_tol: float

class TraceData(NamedTuple):
    trace: np.ndarray # size == (len(decays),)
    quality: np.ndarray # size == (len(decays),)
    obs: np.ndarray # size == (n,)

class TraceConstructor:
    def __init__(self, cfg: TraceConfig):
        self._cfg = cfg
        self._trace_params = TraceParams(
            decays=np.array(cfg.trace_values, dtype=np.float32),
            missing_tol=cfg.missing_tol,
        )

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, TraceTemporalState | None)
        trace = (ts and ts.trace) or {}
        quality = (ts and ts.quality) or {}

        cols = set(carry.transform_data.columns)
        for col in cols:
            obs = carry.transform_data[col].to_numpy(dtype=np.float32)
            prev_trace = Maybe(trace.get(col)).or_else(np.ones_like(self._trace_params.decays, dtype=np.float32)*np.nan)
            prev_quality = Maybe(quality.get(col)).or_else(np.zeros_like(self._trace_params.decays, dtype=np.float32))
            assert obs.ndim == 1, f"shape of column {col}: {obs.shape}, transform_data: {carry.transform_data}"

            trace_data = TraceData(
                trace=prev_trace,
                quality=prev_quality,
                obs=obs,
            )
            out, new_trace_data = compute_trace_with_nan(self._trace_params, trace_data)

            trace[col] = new_trace_data.trace
            quality[col] = new_trace_data.quality
            carry.transform_data.drop(col, axis=1, inplace=True)

            for i, decay in enumerate(self._cfg.trace_values):
                new_name = f'{col}_trace-{decay}'
                carry.transform_data[new_name] = out[:, i]


        return carry, TraceTemporalState(trace, quality)

    def reset(self) -> None:
        pass


transform_group.dispatcher(TraceConstructor)

@njit
def compute_trace_with_nan(trace_params: TraceParams, trace_data: TraceData):
    n_samples = len(trace_data.obs)
    n_traces = len(trace_params.decays)
    out = np.zeros((n_samples, n_traces), dtype=np.float32)

    decays = trace_params.decays
    trace = trace_data.trace
    quality = trace_data.quality

    for i in range(n_samples):
        obs = trace_data.obs[i]

        # initialize trace if needed
        trace_nanmask = np.isnan(trace)
        trace = np.where(trace_nanmask, obs, trace)

        # update trace if obs is not nan, otherwise forward old trace
        obs_not_nan = ~np.isnan(obs)
        trace = decays * trace + (1 - decays) * obs if obs_not_nan else trace
        quality = decays * quality + (1 - decays) * obs_not_nan

        # output trace value
        high_quality = quality > (1 - trace_params.missing_tol)
        out[i] = np.where(high_quality, trace, np.nan)

    return out, TraceData(trace, quality, trace_data.obs)

def log_trace_quality(app_state: AppState, prefix: str, decays: list[float], trace_ts: TraceTemporalState):
    for tag, quality in Maybe(trace_ts.quality).expect().items():
        for i, decay in enumerate(decays):
            metric_name = f'{prefix}-QUALITY-{tag}_trace_{decay}'
            app_state.metrics.write(app_state.agent_step, metric=metric_name, value=quality[i])
