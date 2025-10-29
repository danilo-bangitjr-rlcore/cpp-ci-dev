from dataclasses import dataclass, field
from typing import Any, Protocol

import chex
import jax

from lib_agent.buffer.datatypes import Transition
from lib_agent.critic.critic_utils import CriticMetrics, CriticState, RollingResetConfig


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


class Critic(Protocol):
    """Protocol defining the interface for critic implementations."""

    def __init__(self, cfg: CriticConfig, seed: int, state_dim: int, action_dim: int):
        ...

    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array) -> CriticState:
        """Initialize the critic parameters."""
        ...

    def forward(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        only_active: bool = True,
    ) -> CriticOutputs:
        """Get action-values from the critic ensemble, optionally filtering to active members only."""
        ...

    def get_rolling_reset_metrics(self, prefix: str = "") -> dict[str, float]:
        """Get metrics related to rolling reset functionality."""
        ...

    def get_representations(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array) -> jax.Array:
        """Get internal representations from the critic."""
        ...

    def update(
        self,
        critic_state: CriticState,
        transitions: Transition,
        *args: Any,
    ) -> tuple[CriticState, CriticMetrics]:
        """Update the critic networks using the passed state transitions.

        Note: Different critic implementations may require different additional arguments:
        - AdvCritic requires: policy_actions, policy_probs
        - QRCCritic requires: next_actions
        """
        ...

    def initialize_to_nominal_action(
        self,
        rng: chex.PRNGKey,
        critic_state: CriticState,
        nominal_action: jax.Array,
    ) -> CriticState:
        """Initialize the critic to be maximized at a nominal action."""
        ...
