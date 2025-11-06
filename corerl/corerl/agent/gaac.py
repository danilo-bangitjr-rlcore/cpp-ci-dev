from functools import partial

import chex
import jax
import lib_utils.jax as jax_u
from lib_agent.buffer.datatypes import State, Transition
from lib_agent.critic.adv_critic import AdvCritic
from lib_agent.critic.critic_utils import create_ensemble_dict, extract_metrics, get_stable_rank

from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.agent.greedy_ac import AdvCriticConfig


class GAAC(GreedyAC):
    """Greedy Advantage Actor-Critic (GAAC).

    Inherits from GreedyAC and uses an advantage-based critic.
    """

    @jax_u.method_jit
    def _get_action_probs(self, states: State, actions: jax.Array):
        """Computes action probabilities for advantage centering."""
        chex.assert_rank(states.features, 3)  # (ens, batch, state_dim)
        chex.assert_rank(actions, 4)  # (ens, batch, n_samples, action_dim)

        f = partial(self._actor.get_probs, self._actor_state.actor.params)
        return jax_u.multi_vmap(f, levels=2)(states, actions)

    def _get_actions_and_probs(self, states: State):
        """Samples actions from policy for current state."""
        assert isinstance(self.cfg.critic, AdvCriticConfig)
        num_policy_actions = self.cfg.critic.num_policy_actions
        ensemble_size = self.cfg.critic.critic_network.ensemble
        batch_size = states.features.shape[1]

        self._jax_rng, bs_rng = jax.random.split(self._jax_rng)
        policy_actions, _ = self._actor.get_actions_rng(
            self._actor_state.actor.params,
            bs_rng,
            states,
            num_policy_actions,
        )
        chex.assert_shape(policy_actions, (ensemble_size, batch_size, num_policy_actions, self.action_dim))

        policy_probs = self._get_action_probs(states, policy_actions)
        policy_probs /= policy_probs.sum(axis=-1, keepdims=True)

        chex.assert_shape(policy_probs, (ensemble_size, batch_size, num_policy_actions))
        return policy_actions, policy_probs

    def update_critic(self) -> list[float]:
        """Update critic using advantage-based loss with policy action sampling."""
        assert isinstance(self.critic, AdvCritic)
        if not self.critic_buffer.is_sampleable:
            return [0 for _ in range(len(self.critic._reset_manager.active_indices))]

        critic_batch: Transition = self.critic_buffer.sample()

        # Sample policy actions from current state (not next state)
        policy_actions, policy_probs = self._get_actions_and_probs(critic_batch.state)

        self._jax_rng, critic_update_rng = jax.random.split(self._jax_rng)
        self._critic_state, metrics = self.critic.update(
            seed=critic_update_rng,
            critic_state=self._critic_state,
            transitions=critic_batch,
            policy_actions=policy_actions,
            policy_probs=policy_probs,
        )

        rolling_reset_metrics = self.critic.get_rolling_reset_metrics()

        # Extract AdvCritic-specific metrics
        metrics_dict = create_ensemble_dict(
            metrics,
            lambda m: extract_metrics(m, [
                'layer_grad_norms', 'layer_weight_norms', 'loss', 'v_loss', 'h_loss', 'adv_loss',
                'v', 'h', 'adv', 'delta', 'h_reg_loss',
                'ensemble_grad_norms', 'ensemble_weight_norms',
            ]),
        )
        self._app_state.metrics.write_dict(
            metrics_dict,
            agent_step=self._app_state.agent_step,
        )
        self._app_state.metrics.write_dict(
            rolling_reset_metrics,
            agent_step=self._app_state.agent_step,
        )

        stable_ranks_dict = create_ensemble_dict(
            self._critic_state.params,
            get_stable_rank,
            prefix='stable_rank_',
        )
        self._app_state.metrics.write_dict(
            stable_ranks_dict,
            agent_step=self._app_state.agent_step,
        )

        rolling_reset_metrics = self.critic.get_rolling_reset_metrics()
        self._app_state.metrics.write_dict(
            rolling_reset_metrics,
            agent_step=self._app_state.agent_step,
        )

        return [metrics.loss[i].mean().item() for i in self.critic._reset_manager.active_indices]
