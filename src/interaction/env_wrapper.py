from typing import Any

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
        self.last_state = None

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
        state, self.current_trace_state = self.state_constructor(observation)
        self.last_state = state
        return state, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any], list[Transition]]:
        # states are indexed by the observation space
        # e.g. {'0': observation_0,
        #       '1': observation_1,
        #       '0_trace-0.5': observation_0_trace_value,
        #       '1_trace-0.5': observation_1_trace_value}
        observation, reward, terminated, truncated, info = self.env.step(action)
        state, self.current_trace_state = self.state_constructor(observation)

        done = terminated or truncated
        transitions = self.transition_creator(
            state=self.last_state,
            action=action,
            reward=float(reward),
            next_state=state,
            done=done
        )

        self.last_state = state
        return state, float(reward), terminated, truncated, info, transitions

    def denormalize(self, normalized_observation: dict[str, Any]) -> dict[str, Any]:
        return self.state_constructor.denormalize(normalized_observation)

    def get_state_dim(self) -> int | None:
        return self.state_constructor.get_state_dim()

    def to_array(self, state_dict: dict[str, Any]) -> jnp.ndarray:
        numpy_array = self.state_constructor.to_array(state_dict)
        return jnp.asarray(numpy_array)
