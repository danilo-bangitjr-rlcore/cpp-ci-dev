from dataclasses import dataclass, field
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
    setpoint_schedule: dict = field(default_factory=lambda:{0: 0.5})
    delta_schedule: dict = field(default_factory=lambda:{0: 0.})
    anchor_schedule: dict = field(default_factory=lambda:{0: 0.})

class Saturation(gym.Env):
    def __init__(self, cfg: SaturationConfig):
        if cfg is None:
            cfg = SaturationConfig()

        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = np.zeros(3) # for [saturation, delta, anchor]
        self._obs_max = np.ones(3)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_dim = 1
        self._action_min = np.array([0])
        self._action_max = np.array([0.5])
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self.time_step = 0
        self.saturation = 0.

        self.setpoint_schedule = cfg.setpoint_schedule
        if 0 not in self.setpoint_schedule:
            raise AssertionError("setpoint schedule must start at 0")
        self.saturation_sp = self.setpoint_schedule[0]

        self.delta_schedule = cfg.delta_schedule
        if 0 not in self.delta_schedule:
            raise AssertionError("delta schedule must start at 0")
        self.delta = self.delta_schedule[0]

        self.anchor_schedule = cfg.anchor_schedule
        if 0 not in self.anchor_schedule:
            raise AssertionError("delta schedule must start at 0")
        self.anchor = self.anchor_schedule[0]

        self.decay = cfg.decay
        self.effect = cfg.effect
        self.decay_period = cfg.decay_period
        self.effect_period = cfg.effect_period
        self.action_trace = np.array([0])
        self.trace_val = cfg.trace_val

        # for plotting
        self.saturations = []
        self.decays = []
        self.effects = []
        self.actions = []
        self.raw_actions = []
        self.deltas = []
        self.anchors = []
        self.saturation_sps = []

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
        # adjust scheduled attributes
        if self.time_step in self.setpoint_schedule:
            self.saturation_sp = self.setpoint_schedule[self.time_step]

        if self.time_step in self.delta_schedule:
            self.delta = self.delta_schedule[self.time_step]

        if self.time_step in self.anchor_schedule:
            self.anchor = self.anchor_schedule[self.time_step]

        self.action_trace = self.trace_val * self.action_trace + (1 - self.trace_val) * action
        decay =  self.get_decay()
        effect = self.get_effect()

        self.saturation = self.saturation*decay + self.action_trace*effect
        self.saturation = np.clip(self.saturation, 0, 1).item()
        reward = -np.abs(self.saturation - self.saturation_sp).item()

        self.decays.append(decay)
        self.effects.append(effect)
        self.saturations.append(self.saturation)
        self.raw_actions.append(action)
        self.actions.append(self.action_trace)
        self.deltas.append(self.delta)
        self.anchors.append(self.anchor)
        self.saturation_sps.append(self.saturation_sp)

        self.time_step += 1

        state = np.array([self.saturation, self.delta, self.anchor])

        return state, reward, False, False, {}

    def plot(self, save_path: Path):
        import matplotlib.pyplot as plt
        plt.plot(self.raw_actions, label="raw actions")
        plt.plot(self.actions, label="actions")
        plt.plot(self.saturations, label="saturation")
        plt.plot(self.effects, label="effects")
        plt.plot(self.decays, label="decay")
        plt.plot(self.saturation_sps, label="saturation sp")
        plt.plot(self.deltas, label="delta")
        plt.plot(self.anchors, label="anchor")
        plt.legend()
        plt.savefig(save_path / 'env.png', bbox_inches='tight')
        # plt.show()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.time_step = 0
        self.saturation_sp = self.setpoint_schedule[0]
        self.delta = self.delta_schedule[0]
        self.anchor = self.anchor_schedule[0]
        state = np.array([self.saturation, self.delta, self.anchor])
        return state, {}

    def close(self):
        pass

env_group.dispatcher(SaturationConfig(), Saturation)
