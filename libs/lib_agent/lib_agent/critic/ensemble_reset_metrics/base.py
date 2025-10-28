from abc import ABC, abstractmethod
from typing import Any

import jax
from lib_config.config import MISSING, config

from lib_agent.actor.actor_protocol import Actor, PolicyState
from lib_agent.buffer.datatypes import Transition
from lib_agent.critic.critic_protocol import Critic
from lib_agent.critic.critic_utils import CriticState


@config()
class BaseEnsembleResetMetricConfig:
    name: Any = MISSING
    enabled: bool = True


class BaseEnsembleResetMetric(ABC):
    def __init__(self, config: BaseEnsembleResetMetricConfig, gamma: float):
        self.config = config
        self.gamma = gamma
        self._jax_rng = jax.random.PRNGKey(0)

    @abstractmethod
    def __call__(
        self,
        transition: Transition,
        critic_state: CriticState,
        critic: Critic,
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
