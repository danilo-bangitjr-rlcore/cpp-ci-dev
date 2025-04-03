from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env

from interaction.state_constructor import StateConstructor
from interaction.transition_creator import Transition, TransitionCreator


class EnvWrapper:
    def __init__(
        self,
        env: Env,
        action_space_info: dict[str, Any],
        observation_space_info: dict[str, Any],
        trace_values: Sequence[float] = (0, 0.75, 0.9, 0.95),
        min_n_step: int = 1,
        max_n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.env = env
        self.state_constructor = StateConstructor(
            action_space_info=action_space_info,
            observation_space_info=observation_space_info,
            trace_values=trace_values,
        )
        self.transition_creator = TransitionCreator(
            min_n_step=min_n_step,
            max_n_step=max_n_step,
            gamma=gamma,
        )

        self.current_trace_state = None
        self.last_state = np.zeros(self.state_constructor.get_state_dim())
        self.is_dict_space = hasattr(env.observation_space, 'spaces')

    def reset(self):
        observation, info = self.env.reset()
        assert isinstance(observation, np.ndarray)
        state = self.state_constructor(observation, None)
        self.last_state = state
        return self.last_state, info

    def step(self, action: jax.Array) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], list[Transition]]:
        observation, reward, terminated, truncated, info = self.env.step(np.array(action))
        state = self.state_constructor(observation, np.asarray(action))
        done = terminated or truncated

        transitions = self.transition_creator(
            state=self.last_state,
            action=np.asarray(action),
            reward=float(reward),
            done=done
        )

        self.last_state = state
        return state, float(reward), terminated, truncated, info, transitions

    def denormalize(self, normalized_observation: jax.Array) -> jax.Array:
        low = jnp.array(self.state_constructor.obs_low)
        range_val = jnp.array(self.state_constructor.range)
        return normalized_observation * range_val + low

    def get_state_dim(self) -> int:
        return self.state_constructor.get_state_dim()
