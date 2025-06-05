from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from gymnasium import Env
from ml_instrumentation.Collector import Collector

from interaction.state_constructor import StateConstructor


class EnvWrapper:
    def __init__(
        self,
        env: Env,
        action_space_info: dict[str, Any],
        observation_space_info: dict[str, Any],
        collector: Collector,
        trace_values: Sequence[float] = (0, 0.75, 0.9, 0.95),
        goal_constructor: Any = None,
    ):
        self.env = env
        self._collector = collector
        self.state_constructor = StateConstructor(
            action_space_info=action_space_info,
            observation_space_info=observation_space_info,
            trace_values=trace_values,
        )
        self.current_trace_state = None
        self.is_dict_space = hasattr(env.observation_space, 'spaces')
        self._goal_constructor = goal_constructor

    def reset(self):
        observation, info = self.env.reset()
        if type(observation) is list:
            observation = np.array([float(x) if isinstance(x, np.ndarray | np.float64) else x for x in observation])
        assert isinstance(observation, np.ndarray)
        state = self.state_constructor(observation, None)
        return state, info

    def step(self, action: jax.Array) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = self.state_constructor.denormalize_action(np.array(action))

        for i, ra in enumerate(raw_action):
            self._collector.collect(f'raw_action_{i}', float(ra))

        observation, env_reward, terminated, truncated, info = self.env.step(raw_action)
        # this is for weird observations given out by saturation_goal etc,
        # where the observation can have mixed types [np.array, np.float]
        if type(observation) is list:
            observation = np.array([float(x) if isinstance(x, np.ndarray | np.float64) else x for x in observation])
        reward = env_reward
        if self._goal_constructor is not None:
            reward = self._goal_constructor(observation)

        state = self.state_constructor(observation, raw_action)
        return state, float(reward), terminated, truncated, info

    def get_state_dim(self) -> int:
        return self.state_constructor.get_state_dim()
