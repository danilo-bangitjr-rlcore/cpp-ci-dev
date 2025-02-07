from dataclasses import field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from corerl.configs.config import config


@config()
class MultiActionSaturationConfig:
    effect_period: float = 100
    decay: float = 0.75
    trace_val: float = 0.9
    num_controllers: int = 3
    noise_std: float = 0.00
    coupling_matrix: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])


class MultiActionSaturation(gym.Env):
    """Multi-Action Saturation Environment

    This environment simulates a multi-controller system where each controller tries to maintain
    its target saturation level while dealing with various dynamic effects and interactions.

    Dynamics:
        - Multiple controllers (default: 3) operate simultaneously
        - Each controller has its own target setpoint [0.3, 0.5, 0.7]
        - System state evolves based on:
            1. Base periodic effects (different frequencies per controller)
                - Each controller has different frequency (1.0, 1.5, 2.0)
                - Different phase shifts (0, π/4, π/2)
                - Amplitude: 0.15 * cos(.) + 0.75
            3. Exponential smoothing of actions
            4. Random noise
    """
    def __init__(self, cfg: MultiActionSaturationConfig, seed: int | None = None):
        self._random = np.random.default_rng(seed)
        self.num_controllers = cfg.num_controllers

        self._obs_min = np.zeros(self.num_controllers)
        self._obs_max = np.ones(self.num_controllers)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max)

        self.saturations = np.zeros(self.num_controllers)
        self.setpoints = np.array([0.3, 0.5, 0.7])[:self.num_controllers]
        self.effect_period = cfg.effect_period
        self.noise_std = cfg.noise_std

        self._action_min = np.zeros(self.num_controllers)
        self._action_max = np.ones(self.num_controllers)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max)

        self.time_step = 0
        self.decay = cfg.decay

        self.history_saturations = []
        self.history_effects = []
        self.history_actions = []
        self.history_raw_actions = []
        self.action_traces = np.zeros(self.num_controllers)
        self.trace_val = cfg.trace_val

        self.frequencies = np.array([1.0, 1.5, 2.0])[:self.num_controllers]
        self.phase_shifts = np.array([0, np.pi/4, np.pi/2])[:self.num_controllers]

        self.coupling_matrix = np.array(cfg.coupling_matrix[:self.num_controllers])[:, :self.num_controllers]

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def step(self, action: np.ndarray):
        self.time_step += 1

        self.action_traces = self.trace_val * self.action_traces + (1 - self.trace_val) * action
        self.saturations = self.saturations * self.decay

        base_effects = np.zeros(self.num_controllers)
        for i in range(self.num_controllers):
            base_effects[i] = 0.15 * np.cos(
                self.time_step * np.pi * (2 / self.effect_period) *
                self.frequencies[i] + self.phase_shifts[i]
            ) + 0.75

        noise = self._random.normal(0, self.noise_std, self.num_controllers)

        coupled_effects = np.dot(self.coupling_matrix, self.saturations)

        self.saturations = coupled_effects + (
            self.action_traces * base_effects +
            noise +
            coupled_effects
        )
        self.saturations = np.clip(self.saturations, 0, 1)

        errors = self.saturations - self.setpoints
        reward = -np.sum(np.abs(errors))

        self.history_effects.append(base_effects)
        self.history_saturations.append(self.saturations.copy())
        self.history_raw_actions.append(action.copy())
        self.history_actions.append(self.action_traces.copy())

        return self.saturations, reward, False, False, {}

    def plot(self, save_path: Path):
        import matplotlib.pyplot as plt
        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        history_saturations = np.array(self.history_saturations)
        history_actions = np.array(self.history_actions)
        history_effects = np.array(self.history_effects)

        for i in range(self.num_controllers):
            color = f'C{i}'
            difference = np.abs(history_saturations[:, i] - self.setpoints[i])
            ax1.plot(difference, color=color, label=f"difference to setpoint {i}")
            ax1.plot(history_effects[:, i], color=color, linestyle=':', label=f"effect {i}")
        ax1.legend()
        ax1.set_title("Different to setpoint and Effects")

        for i in range(self.num_controllers):
            color = f'C{i}'
            ax2.plot(history_actions[:, i], color=color, label=f"action trace {i}")
        ax2.legend()
        ax2.set_title("Action Traces")

        plt.savefig(save_path / 'env.png', bbox_inches='tight')
        plt.close()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._random = np.random.default_rng(seed)

        self.saturations = np.zeros(self.num_controllers)
        self.action_traces = np.zeros(self.num_controllers)
        self.time_step = 0

        self.history_saturations = []
        self.history_effects = []
        self.history_actions = []
        self.history_raw_actions = []

        obs = np.concatenate([self.saturations, self.saturations - self.setpoints])
        return obs, {}

    def close(self):
        pass

gym.register(
    id='MultiActionSaturation-v0',
    entry_point='corerl.environment.multi_action_saturation:MultiActionSaturation',
    kwargs={'cfg': MultiActionSaturationConfig()}
)

