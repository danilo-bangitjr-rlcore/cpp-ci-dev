from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from coreenv.factory import EnvConfig, env_group


@dataclass
class CalibrationConfig(EnvConfig):
    name: str = 'Calibration-v0'
    seed: int = 0
    calibration_period: int = 500

class CalibrationEnv(gym.Env):
    """
    Environment where the agent must trade off between "cost" and "efficiency".
    The cost is equal to the action on any given step, and efficiency is a linear function
    of the action. The intended reward is:
        (1st priority) maintain efficiency above 0.5
        (2nd priority) minimize cost
    which can be easiliy implemented with the goal constructor.

    Every "calibration_period" steps, the bias term in the linear function for computing efficiency
    changes. The agent should quickly adapt its policy. This change in the bias term makes the environment
    partially observable if no traces are used in the state constructor. If traces are enabled, the environment
    is fully observable.
    """
    def __init__(self, cfg:  CalibrationConfig):
        self._cfg = cfg
        self._random = np.random.default_rng(self._cfg.seed)
        self._obs_min = np.zeros(2)
        self._obs_max = np.ones(2)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_min = np.zeros(1)
        self._action_max = np.zeros(1)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self._scale = 0.7
        self._bias = 0.3
        self._calibration_period = self._cfg.calibration_period
        self._steps = 0


    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cost = action
        efficiency = action * self._scale + self._bias

        self.state = np.concatenate([cost, efficiency])
        reward = 0. # this reward is ignored, goal constructor should be used instead
        self._steps += 1

        if (self._steps // self._calibration_period) % 2 == 1:
            # optimal action is ~0.571
            self._bias = 0.1
        else:
            # optimal action is ~0.286
            self._bias = 0.3

        return self.state, reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._steps = 0
        action = np.ones(1) * 0.5
        cost = action
        efficiency = action * self._scale + self._bias

        self.state = np.concatenate([cost, efficiency])

        return self.state, {}

    def close(self):
        pass

env_group.dispatcher(CalibrationConfig(),CalibrationEnv)
