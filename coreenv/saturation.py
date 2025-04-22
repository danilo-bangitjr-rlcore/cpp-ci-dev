from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from coreenv.factory import EnvConfig, env_group


@dataclass
class SaturationConfig(EnvConfig):
    name: str = 'Saturation-v0'
    effect_period: float = 100
    decay_period: float = 100
    # None signifies following a cosine wave
    decay: float | None = None
    effect: float | None = 1.0
    trace_val: float = 0.0

class Saturation(gym.Env):
    def __init__(self, cfg: SaturationConfig):
        if cfg is None:
            cfg = SaturationConfig()

        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = np.array([0.])
        self._obs_max = np.array([1.])
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([0.5])
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self.time_step = 0
        self.saturation = np.array([0.])
        self.saturation_sp = np.array([0.5])

        self.decay = cfg.decay
        self.effect = cfg.effect
        self.decay_period = cfg.decay_period
        self.effect_period = cfg.effect_period

        self.saturations = []
        self.decays = []
        self.effects = []
        self.actions = []
        self.raw_actions = []
        self.action_trace = np.array([0])
        self.trace_val = cfg.trace_val

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def get_decay(self) -> float | np.ndarray:
        if self.decay is None:
            return 0.15 * np.cos(self.time_step * np.pi * (2 / self.decay_period)) + 0.75
        else:
            return self.decay

    def get_effect(self)-> float | np.ndarray:
        if self.effect is None:
            return 0.15 * np.cos(self.time_step * np.pi * (2 / self.effect_period)) + 0.75
        else:
            return self.effect

    def step(self, action: np.ndarray):
        self.time_step += 1

        self.action_trace = self.trace_val * self.action_trace + (1 - self.trace_val) * action
        decay =  self.get_decay()
        effect = self.get_effect()

        self.saturation = self.saturation*decay + self.action_trace*effect
        self.saturation = np.clip(self.saturation, 0, 1)
        reward = -np.abs(self.saturation - self.saturation_sp).item() # type: ignore

        self.decays.append(decay)
        self.effects.append(effect)
        self.saturations.append(self.saturation)
        self.raw_actions.append(action)
        self.actions.append(self.action_trace)

        return self.saturation, reward, False, False, {}

    def plot(self, save_path: Path):
        import matplotlib.pyplot as plt
        plt.plot(self.raw_actions, label="raw actions")
        plt.plot(self.actions, label="actions")
        plt.plot(self.saturations, label="saturation")
        plt.plot(self.effects, label="effects")
        plt.legend()
        plt.savefig(save_path / 'env.png', bbox_inches='tight')
        # plt.show()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        return self.saturation, {}

    def close(self):
        pass

env_group.dispatcher(SaturationConfig(), Saturation)
