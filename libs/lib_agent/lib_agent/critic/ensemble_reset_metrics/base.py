from abc import ABC, abstractmethod
from typing import Any, Protocol

import chex
import jax
from lib_config.config import MISSING, config

from lib_agent.actor.actor_protocol import Actor, PolicyState
from lib_agent.buffer.datatypes import Transition
from lib_agent.critic.critic_utils import CriticOutputs, CriticState


class EnsembleResetMetricCritic(Protocol):
    """Protocol defining the interface for critics used in BaseEnsembleResetMetric."""

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


@config()
class BaseEnsembleResetMetricConfig:
    name: Any = MISSING
    enabled: bool = True


class BaseEnsembleResetMetric(ABC):
    def __init__(self, cfg: BaseEnsembleResetMetricConfig, gamma: float):
        self.config = cfg
        self.gamma = gamma

    @abstractmethod
    def __call__(
        self,
        rng: chex.PRNGKey,
        transition: Transition,
        critic_state: CriticState,
        critic: EnsembleResetMetricCritic,
        actor_state: PolicyState,
        actor: Actor,
    ) -> jax.Array:
        """
        Calculate ensemble reset metric from a transition.

        Args:
            transition: The transition data containing state, action, reward, etc.
            critic_state: The parameters of the critics in the ensemble.
            critic: The critic algorithm instance.
            actor_state: The parameters of the actor.
            actor: The actor algorithm instance.

        Returns:
            A jax Array containing the ensemble reset metric values for each critic.
        """
        raise NotImplementedError
