from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TraceState:
    mu: dict[str, np.ndarray | None] = field(default_factory=dict)

class StateConstructor:
    def __init__(
        self,
        observation_space_info: dict[str, Any] | None = None,
        trace_values: list[float] | None = None
    ):
        self.normalize = observation_space_info is not None
        self.trace = trace_values is not None
        self.keys_order = None
        self.obs_low = None
        self.obs_high = None
        self.range = None

        if observation_space_info is not None:
            self.obs_low = observation_space_info['low']
            self.obs_high = observation_space_info['high']
            self.range = self.obs_high - self.obs_low

        if self.trace:
            self._decays = np.array(trace_values)

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        state: TraceState | None = None
    ) -> tuple[dict[str, np.ndarray], TraceState | None]:
        result = {}

        if self.normalize:
            for key, value in observation.items():
                result[key] = (value - self.obs_low) / self.range
        else:
            result = observation.copy()

        if self.trace:
            if state is None or state.mu is None:
                mu = {}
            else:
                mu = state.mu

            trace_result = {}
            for col, x in (result if self.normalize else observation).items():
                trace_vals, new_mu = self._compute_trace(
                    data=x,
                    decays=self._decays,
                    mu_0=mu.get(col)
                )

                mu[col] = new_mu
                for i, decay in enumerate(self._decays):
                    new_name = f'{col}_trace-{decay}'
                    trace_result[new_name] = trace_vals[:, i]

            result.update(trace_result)
            return result, TraceState(mu)

        return result, None

    def denormalize(self, normalized_observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.normalize:
            return normalized_observation

        result = {}
        for key, value in normalized_observation.items():
            if not key.startswith('_trace-'):
                result[key] = value * self.range + self.obs_low
            else:
                result[key] = value
        return result

    def _compute_trace(self, data: np.ndarray, decays: np.ndarray, mu_0: np.ndarray | None = None):
        n_samples = len(data)
        n_traces = len(decays)
        out = np.zeros((n_samples, n_traces), dtype=np.float64)
        mu = mu_0

        for i in range(n_samples):
            x = data[i]
            if np.isnan(x):
                mu = None
                out[i] = np.nan
                continue

            if mu is None:
                mu = np.ones(n_traces, dtype=np.float64) * x

            mu = decays * mu + (1 - decays) * x
            out[i] = mu

        return out, mu

    def get_state_dim(self) -> int | None:
        if not self.normalize or self.obs_low is None:
            return None

        base_dim = len(self.obs_low)
        if not self.trace:
            return base_dim

        trace_multiplier = len(self._decays)
        return base_dim * (trace_multiplier + 1)

    def to_array(self, state_dict: dict[str, np.ndarray]) -> np.ndarray:
        if self.keys_order is None:
            self.keys_order = []
            for key in sorted(state_dict.keys()):
                if '_trace-' not in key:
                    self.keys_order.append(key)
            if self.trace:
                for base_key in self.keys_order.copy():
                    for decay in self._decays:
                        trace_key = f'{base_key}_trace-{decay}'
                        if trace_key in state_dict:
                            self.keys_order.append(trace_key)

        arrays = [state_dict[key] for key in self.keys_order]
        return np.stack(arrays, axis=-1)
