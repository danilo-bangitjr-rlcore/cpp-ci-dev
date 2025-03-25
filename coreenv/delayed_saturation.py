from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from corerl.configs.config import computed, config, list_
from corerl.configs.loader import config_from_dict

if TYPE_CHECKING:
    from corerl.config import MainConfig

@config()
class DelayedSaturationConfig:
    effect_period: float = 500
    decay: float = 0.75
    trace_val: float = 0.9
    action_names: list[str] = list_()
    endo_inds: list[int] = list_()
    endo_obs_names: list[str] = list_()
    seed: int = 1

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed



class DelayedSaturation(gym.Env):
    def __init__(self, cfg: DelayedSaturationConfig | None = None):
        if cfg is None:
            cfg = DelayedSaturationConfig()

        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = np.array([0.])
        self._obs_max = np.array([1.])
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self.saturation = np.array([0.5])
        self.saturation_sp = np.array([0.5])
        self.effect_period = cfg.effect_period

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([1])
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self.time_step = 0
        self.decay = cfg.decay

        self.saturations = []
        self.effects = []
        self.actions = []
        self.raw_actions = []
        self.action_trace = np.array([0])
        self.trace_val = cfg.trace_val

        self.action_names = cfg.action_names
        self.endo_inds = cfg.endo_inds
        self.endo_obs_names = cfg.endo_obs_names

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def step(self, action: np.ndarray):
        self.time_step += 1

        self.action_trace = self.trace_val * self.action_trace + (1 - self.trace_val) * action
        self.saturation = self.saturation * self.decay
        effect = 0.15 * np.cos(self.time_step * np.pi * (2 / self.effect_period)) + 0.75
        self.saturation += self.action_trace * effect
        self.saturation = np.clip(self.saturation, 0, 1)
        reward = -np.abs(self.saturation - self.saturation_sp).item()

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

gym.register(
    id='DelayedSaturation-v0',
    entry_point='corerl.environment.delayed_saturation:DelayedSaturation'
)
