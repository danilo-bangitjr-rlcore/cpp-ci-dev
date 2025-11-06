import chex
import jax
import jax.numpy as jnp
from lib_config.config import config

from lib_agent.actor.actor_protocol import Actor, PolicyState
from lib_agent.buffer.datatypes import Transition
from lib_agent.critic.critic_protocol import Critic
from lib_agent.critic.critic_utils import CriticState
from lib_agent.critic.ensemble_reset_metrics.base import BaseEnsembleResetMetric, BaseEnsembleResetMetricConfig


@config()
class DivergenceMetricConfig(BaseEnsembleResetMetricConfig):
    name: str = "divergence"
    tolerance: float = 1.0
    num_actions: int = 10
    min_reward: float = -8.0
    max_reward: float = 0.0


class DivergenceMetric(BaseEnsembleResetMetric):
    def __init__(self, cfg: DivergenceMetricConfig, gamma: float):
        super().__init__(cfg, gamma)
        self.num_actions = cfg.num_actions
        self.min_return = (cfg.min_reward / (1 - gamma)) - cfg.tolerance
        self.max_return = (cfg.max_reward / (1 - gamma)) + cfg.tolerance
        self.return_range = self.max_return - self.min_return

    def __call__(
        self,
        rng: chex.PRNGKey,
        transition: Transition,
        critic_state: CriticState,
        critic: Critic,
        actor_state: PolicyState,
        actor: Actor,
    ) -> jax.Array:
        """
        Calculate divergence score for each critic in the ensemble at the state in the given transition.

        Each critic is evaluated over a set of uniformly sampled actions within the action bounds at the given state.
        The average difference of the action-values from the min/max possible returns is computed for each critic.
        The divergence metric for each critic is the average difference normalized by (max_return - min_return).
        """
        # Sample actions over which to evaluate ensemble of critics at the given state
        action_dim = transition.action.shape[-1]
        a_los = transition.state.a_lo
        a_his = transition.state.a_hi
        _, key = jax.random.split(rng)
        uniform_actions = jax.random.uniform(key=key, minval=a_los, maxval=a_his, shape=(self.num_actions, action_dim))

        # Get action-values from each critic for the sampled actions
        q_ens = critic.forward(
            critic_state.params,
            rng,
            transition.state.features.array,
            uniform_actions,
            only_active=False,
        ).q

        # If values are greater than max_return, diff is computed with respect to max_return
        # If values are less than min_return, diff is computed with respect to min_return
        clamped_q = jnp.clip(q_ens, self.min_return, self.max_return)
        diffs = jnp.abs(q_ens - clamped_q)

        # Metric is computed with respect to the average difference from min/max return over sampled actions
        avg_diffs = jnp.mean(diffs, axis=1)

        return avg_diffs / self.return_range
