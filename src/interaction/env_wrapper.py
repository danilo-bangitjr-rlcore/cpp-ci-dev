from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env

from .state_constructor import StateConstructor
from .transition_creator import Transition, TransitionCreator


class EnvWrapper:
    def __init__(
        self,
        env: Env,
        observation_space_info: dict[str, Any] | None = None,
        trace_values: list[float] | None = None,
        min_n_step: int = 1,
        max_n_step: int = 1,
        gamma: float = 1.0
    ):
        self.env = env
        self.state_constructor = StateConstructor(
            observation_space_info=observation_space_info,
            trace_values=trace_values
        )
        self.transition_creator = TransitionCreator(
            min_n_step=min_n_step,
            max_n_step=max_n_step,
            gamma=gamma
        )

        self.current_trace_state = None
        self.last_state = jnp.zeros(self.state_constructor.get_state_dim())
        self.is_dict_space = hasattr(env.observation_space, 'spaces')

    def _process_observation(self, observation: Any) -> dict[str, Any]:
        if isinstance(observation, dict):
            return observation
        elif isinstance(observation, (np.ndarray, list)):
            return {str(i): val for i, val in enumerate(observation)}
        else:
            return {'0': observation}

    def reset(self):
        observation, info = self.env.reset()
        observation_dict = self._process_observation(observation)
        state_dict, self.current_trace_state = self.state_constructor(observation_dict)
        state_array = self.to_array(state_dict)
        self.last_state = state_array
        return state_array, info

    def step(self, action: jax.Array) -> tuple[jax.Array, float, bool, bool, dict[str, Any], list[Transition]]:
        observation, reward, terminated, truncated, info = self.env.step(np.array(action))
        observation_dict = self._process_observation(observation)
        state_dict, self.current_trace_state = self.state_constructor(observation_dict)
        state_array = self.to_array(state_dict)
        done = terminated or truncated
        
        transitions = self.transition_creator(
            state=self.last_state,
            action=jnp.asarray(action),
            reward=float(reward),
            done=done
        )

        self.last_state = state_array
        return state_array, float(reward), terminated, truncated, info, transitions

    def denormalize(self, normalized_observation: jax.Array) -> jax.Array:
        if not self.state_constructor.normalize:
            return normalized_observation

        low = jnp.array(self.state_constructor.obs_low)
        range_val = jnp.array(self.state_constructor.range)
        return normalized_observation * range_val + low

    def get_state_dim(self) -> int | None:
        return self.state_constructor.get_state_dim()

    def to_array(self, state_dict: dict[str, Any]) -> jax.Array:
        numpy_array = self.state_constructor.to_array(state_dict)
        return jnp.asarray(numpy_array)
