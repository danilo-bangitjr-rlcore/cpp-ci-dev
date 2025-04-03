from collections.abc import Sequence
from typing import Any

import numba
import numpy as np


class StateConstructor:
    def __init__(
        self,
        observation_space_info: dict[str, Any],
        trace_values: Sequence[float],
    ):
        self.obs_low = np.asarray(observation_space_info['low'])
        self.obs_high = np.asarray(observation_space_info['high'])
        self.range = self.obs_high - self.obs_low
        self._decays = np.asarray(trace_values)
        self._mu = np.zeros_like(self._decays)
        self._should_reset = True

    def __call__(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation = _normalize(observation, self.obs_low, self.obs_high)

        if self._should_reset:
            self._mu = observation
            self._should_reset = False

        else:
            self._mu = _update_traces(observation, self._mu, self._decays)

        return np.concatenate([observation, self._mu], axis=0)

    def denormalize(self, normalized_observation: np.ndarray) -> np.ndarray:
        return _denormalize(normalized_observation, self.obs_low, self.obs_high)


    def get_state_dim(self) -> int:
        base = len(self.obs_low)
        return base * (len(self._decays) + 1)

@numba.njit(cache=True)
def _normalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (x - low) / (high - low)


@numba.njit(cache=True)
def _denormalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return x * (high - low) + low


@numba.njit(cache=True)
def _update_traces(x: np.ndarray, mu: np.ndarray, decays: np.ndarray) -> np.ndarray:
    return decays * mu + (1 - decays) * x
