from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env

from interaction.state_constructor import StateConstructor


class EnvWrapper:
    def __init__(
        self,
        env: Env,
        action_space_info: dict[str, Any],
        observation_space_info: dict[str, Any],
        trace_values: Sequence[float] = (0, 0.75, 0.9, 0.95),
    ):
        self.env = env
        self.state_constructor = StateConstructor(
            action_space_info=action_space_info,
            observation_space_info=observation_space_info,
            trace_values=trace_values,
        )
        self.current_trace_state = None
        self.is_dict_space = hasattr(env.observation_space, 'spaces')

    def reset(self):
        observation, info = self.env.reset()
        assert isinstance(observation, np.ndarray)
        state = self.state_constructor(observation, None)
        return state, info

    def step(self, action: jax.Array) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = self.state_constructor.denormalize_action(np.array(action))
        observation, reward, terminated, truncated, info = self.env.step(raw_action)
        state = self.state_constructor(observation, np.asarray(action))
        return state, float(reward), terminated, truncated, info

    def get_state_dim(self) -> int:
        return self.state_constructor.get_state_dim()
