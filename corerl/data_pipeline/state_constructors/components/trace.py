import numpy as np
from numba import njit
from dataclasses import dataclass
from corerl.utils.hydra import list_
from corerl.data_pipeline.state_constructors.interface import TransformCarry
from corerl.data_pipeline.state_constructors.components.base import BaseTransformConfig, sc_group


@dataclass
class TraceConfig(BaseTransformConfig):
    name: str = 'multi_trace'
    trace_values: list[float] = list_([0.9, 0.95])


@dataclass
class TraceTemporalState:
    mu: dict[str, np.ndarray | None] | None = None


class TraceConstructor:
    def __init__(self, cfg: TraceConfig):
        self._cfg = cfg
        self._decays = np.array(cfg.trace_values)

    def __call__(self, carry: TransformCarry, ts: object | None):
        assert isinstance(ts, TraceTemporalState | None)
        mu = ts and ts.mu or {}

        cols = set(carry.agent_state.columns)
        for col in cols:
            x = carry.agent_state[col].to_numpy()

            trace_vals, new_mu = compute_trace_with_nan(
                data=x,
                decays=self._decays,
                mu_0=mu.get(col),
            )

            mu[col] = new_mu
            carry.agent_state.drop(col, axis=1, inplace=True)

            for i in range(len(self._decays)):
                decay = self._decays[i]
                new_name = f'{col}_trace-{decay}'
                carry.agent_state[new_name] = trace_vals[:, i]

        return carry, TraceTemporalState(mu)


sc_group.dispatcher(TraceConstructor)

@njit
def compute_trace_with_nan(
    data: np.ndarray,
    decays: np.ndarray,
    mu_0: np.ndarray | None = None,
):
    n_samples = len(data)
    n_traces = len(decays)
    out = np.zeros((n_samples, n_traces), dtype=np.float_)
    mu: np.ndarray | None = mu_0

    for i in range(n_samples):
        x: float = data[i]
        if np.isnan(x):
            mu = None
            out[i] = np.nan
            continue

        if mu is None:
            mu = np.ones(n_traces, dtype=np.float_) * x

        mu = decays * mu + (1 - decays) * x
        out[i] = mu

    return out, mu
