"""
Gamma schedulers for time-varying discount factors.

NOTE: These schedulers are designed for n=1 (1-step) transitions only.
With n=1, the Bellman equation is: Target = r + γ·V(s')
Since the immediate reward 'r' has no discount applied, we can safely
adjust γ without recomputing rewards.

For n>1 transitions, adjusting gamma would require recomputing n_step_rewards,
which is not supported by these schedulers.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from lib_agent.buffer.datatypes import Transition


class GammaScheduler(ABC):
    def __init__(self, max_gamma: float):
        self.max_gamma = max_gamma

    @abstractmethod
    def get_gamma(self, step: int) -> float:
        ...

    def set_transition_gamma(self, batch: Transition, step: int):
        new_gamma = self.get_gamma(step)
        new_gammas = jnp.full_like(batch.n_step_gamma, new_gamma)

        return batch._replace(n_step_gamma=new_gammas)


class LogarithmicGammaScheduler(GammaScheduler):
    def __init__(
        self,
        max_gamma: float = 0.99,
        update_interval: int = 25,
        horizon: int = 1000,
    ):
        assert 0 <= max_gamma <= 1, f"max_gamma must be in [0, 1], got {max_gamma}"
        assert update_interval > 0, f"update_interval must be positive, got {update_interval}"
        assert horizon > 0, f"horizon must be positive, got {horizon}"

        super().__init__(max_gamma=max_gamma)
        self._update_interval = update_interval
        self._horizon = horizon

    def get_gamma(self, step: int) -> float:
        # only increase step every self._update_interval steps
        step = (step // self._update_interval) * self._update_interval

        # Logarithmic growth reaches max_gamma at step=horizon
        numerator = math.log(step + 1)
        denominator = math.log(self._horizon + 1) / self.max_gamma
        gamma = numerator / denominator
        return min(gamma, self.max_gamma)


@dataclass
class GammaScheduleConfig:
    type: str = "logarithmic"
    max_gamma: float = 0.99
    update_interval: int = 25
    horizon: int = 1000

    def __post_init__(self):
        """Validates configuration parameters."""
        if self.type not in {"logarithmic"}:
            raise ValueError(f"type must be 'logarithmic', got '{self.type}'")

        if not 0 <= self.max_gamma <= 1:
            raise ValueError(f"max_gamma must be in [0, 1], got {self.max_gamma}")

        if self.update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {self.update_interval}")

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

    def to_dict(self):
        """Returns configuration as a dictionary."""
        return {
            "type": self.type,
            "max_gamma": self.max_gamma,
            "update_interval": self.update_interval,
            "horizon": self.horizon,
        }


def create_gamma_scheduler(config: GammaScheduleConfig | None) -> GammaScheduler | None:
    """Creates and returns the appropriate gamma scheduler instance from config."""
    if config is None:
        return None

    if config.type == 'logarithmic':
        return LogarithmicGammaScheduler(
            max_gamma=config.max_gamma,
            update_interval=config.update_interval,
            horizon=config.horizon,
        )
    raise ValueError(f"Unknown scheduler type: {config.type}")
