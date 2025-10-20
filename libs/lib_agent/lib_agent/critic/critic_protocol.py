from dataclasses import dataclass, field
from typing import Any, Protocol

import jax

from lib_agent.critic.critic_utils import CriticBatch, CriticMetrics, CriticState, RollingResetConfig


class CriticOutputs(Protocol):
    q: jax.Array


@dataclass
class CriticConfig:
    name: str
    stepsize: float
    ensemble: int
    ensemble_prob: float
    num_rand_actions: int
    action_regularization: float
    l2_regularization: float
    nominal_setpoint_updates: int = 1000
    use_all_layer_norm: bool = False
    rolling_reset_config: RollingResetConfig = field(default_factory=RollingResetConfig)

