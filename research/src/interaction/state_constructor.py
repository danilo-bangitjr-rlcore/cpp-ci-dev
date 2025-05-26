from collections.abc import Sequence
from typing import Any

import numba
import numpy as np


class StateConstructor:
    def __init__(
        self,
        action_space_info: dict[str, Any],
        observation_space_info: dict[str, Any],
        trace_values: Sequence[float],
    ):
        self.act_low = np.asarray(action_space_info['low'])
        self.act_high = np.asarray(action_space_info['high'])
        self.obs_low = np.asarray(observation_space_info['low'])
        self.obs_high = np.asarray(observation_space_info['high'])
        self.range = self.obs_high - self.obs_low
        self._decays = np.asarray(trace_values)
        self._mu = np.zeros((len(self.obs_low) + len(self.act_low), len(self._decays)))
        self._should_reset = True

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray | None,
    ) -> np.ndarray:
        observation = _normalize(observation, self.obs_low, self.obs_high)

        if action is not None:
            action = _normalize(action, self.act_low, self.act_high)
        else:
            action = np.zeros(self.act_low.shape[0])

        observation = np.concatenate((observation, action), axis=0)

        if self._should_reset:
            self._mu = (
                np.expand_dims(observation, axis=1)
                .repeat(len(self._decays), axis=1)
            )
            self._should_reset = False

        else:
            self._mu = _update_traces(observation, self._mu, self._decays)

        return self._mu.flatten()

    def denormalize_obs(self, normalized_observation: np.ndarray) -> np.ndarray:
        return _denormalize(normalized_observation, self.obs_low, self.obs_high)

    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        return _denormalize(normalized_action, self.act_low, self.act_high)


    def get_state_dim(self) -> int:
        base = len(self.obs_low) + len(self.act_low)
        return base * len(self._decays)

@numba.njit(cache=True)
def _normalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (x - low) / (high - low)


@numba.njit(cache=True)
def _denormalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return x * (high - low) + low


@numba.njit(cache=True)
def _update_traces(x: np.ndarray, mu: np.ndarray, decays: np.ndarray) -> np.ndarray:
    for i in range(x.shape[0]):
        mu[i] = decays * mu[i] + (1 - decays) * x[i]

    return mu
