from collections import deque
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from lib_config.config import config
from pydantic import Field

from rl_env.group_util import EnvConfig, env_group


@config(frozen=True)
class SaturationConfig(EnvConfig):
    name: Literal['Saturation-v0'] = 'Saturation-v0'
    effect_period: float = 100
    decay_period: float = 100
    # None signifies following a cosine wave
    decay: float | None = None
    effect: float | None = 1.0
    trace_val: float = 0.0
    setpoint_schedule: dict = Field(default_factory=lambda: {0: 0.5})
    delta_schedule: dict = Field(default_factory=lambda: {0: 0.})
    anchor_schedule: dict = Field(default_factory=lambda: {0: 0.})
    num_reward_modes: int = 0 # Number of cosine periods in the multimodal reward
    mode_amplitude: float = 0.0 # Amplitude of the cosine components in the multimodal reward
    delay: int = 0 # The number of time steps to delay the saturation observation
    filter_thresh: float | None = None # The observed saturation is the amount above this threshold

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
        self.filter_saturation = 0.

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
        self.filter_thresh = cfg.filter_thresh
        self.decay_period = cfg.decay_period
        self.effect_period = cfg.effect_period
        self.action_trace = np.array([0])
        self.trace_val = cfg.trace_val
        self.num_reward_modes = cfg.num_reward_modes
        self.mode_amplitude = cfg.mode_amplitude
        self.delayed_saturations = deque([0.0]*(cfg.delay+1), maxlen=cfg.delay+1)

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
        return self.decay

    def get_effect(self)-> float | np.ndarray:
        if self.effect is None:
            return 0.15 * np.cos(self.time_step * np.pi * (2 / self.effect_period)) + 0.75
        return self.effect

    def get_multimodal_reward(self, saturation: float, saturation_sp: float) -> float:
        """
        Compute multimodal reward function using cosine waves.
        """
        diff = saturation - saturation_sp

        # Base reward component - highest peak at setpoint
        base_reward = -np.abs(diff)

        # Create multimodal structure using cosine function
        cosine_component = self.mode_amplitude * np.cos(2 * np.pi * self.num_reward_modes * diff)

        return base_reward + cosine_component

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

        self.filter_saturation = self.filter_saturation*decay + self.action_trace*effect
        if self.filter_thresh:
            # Observed saturation is the amount above the filter threshold
            saturation = np.clip(self.filter_saturation - self.filter_thresh, 0, 1).item()
            self.filter_saturation = np.clip(self.filter_saturation, 0.0, self.filter_thresh).item()
            self.delayed_saturations.appendleft(saturation)
        else:
            self.filter_saturation = np.clip(self.filter_saturation, 0, 1).item()
            self.delayed_saturations.appendleft(self.filter_saturation)

        observed_saturation = self.delayed_saturations.pop()
        reward = self.get_multimodal_reward(observed_saturation, self.saturation_sp)

        self.decays.append(decay)
        self.effects.append(effect)
        self.saturations.append(observed_saturation)
        self.raw_actions.append(action)
        self.actions.append(self.action_trace)
        self.deltas.append(self.delta)
        self.anchors.append(self.anchor)
        self.saturation_sps.append(self.saturation_sp)

        self.time_step += 1

        state = np.array([observed_saturation, self.delta, self.anchor])

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
        self.filter_saturation = 0.
        self.action_trace = np.array([0])
        state = np.array([self.filter_saturation, self.delta, self.anchor])
        return state, {}

    def close(self):
        pass

env_group.dispatcher(SaturationConfig(), Saturation)
