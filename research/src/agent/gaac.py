from functools import partial

import chex
import jax
import lib_utils.jax as jax_u
from lib_agent.buffer.datatypes import State
from lib_agent.critic.adv_critic import AdvCritic

from src.agent.gac import GreedyAC


class GAAC(GreedyAC):
    """Greedy Advantage Actor-Critic (GAAC).

    Inherits from GreedyAC and overrides critic_update to sample policy actions
    and compute their probabilities for the advantage centering loss.
    """

    @jax_u.method_jit
    def _get_action_probs(self, states: State, actions: jax.Array):
        chex.assert_rank(states.features, 3)  # (ens, batch, state_dim)
        chex.assert_rank(actions, 4)  # (ens, batch, n_samples, state_dim)

        f = partial(self.get_probs, self.agent_state.actor.actor.params)
        return jax_u.multi_vmap(f, levels=2)(
            states,
            actions,
        )

    def _get_actions_and_probs(self, states: State):
        """Sample actions from policy for current state (for advantage centering)"""
        num_policy_actions = self._cfg.critic.get('num_policy_actions', 100)
        ensemble_size = self._cfg.critic['ensemble']
        batch_size = self._cfg.batch_size

        self.rng, bs_rng = jax.random.split(self.rng)
        policy_actions, _ = self._actor.get_actions_rng(
            self.agent_state.actor.actor.params,
            bs_rng,
            states,
            num_policy_actions,
        )
        chex.assert_shape(policy_actions, (ensemble_size, batch_size, num_policy_actions, self.action_dim))

        policy_probs = self._get_action_probs(
            states,
            policy_actions,
        )
        policy_probs /= policy_probs.sum(axis=-1, keepdims=True)

        chex.assert_shape(policy_probs, (ensemble_size, batch_size, num_policy_actions))
        return policy_actions, policy_probs

    def critic_update(self, step: int):
        assert isinstance(self._critic, AdvCritic)
        if self.critic_buffer.size == 0:
            return 0.

        transitions = self.critic_buffer.sample()

        # Create state for current timestep
        policy_actions, policy_probs = self._get_actions_and_probs(transitions.state)

        self.rng, update_rng = jax.random.split(self.rng)
        # Update critic with policy actions and probabilities
        new_critic_state, metrics = self._critic.update(
            seed=update_rng,
            critic_state=self.agent_state.critic,
            transitions=transitions,
            policy_actions=policy_actions,
            policy_probs=policy_probs,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)
        loss = metrics.loss.mean().item()
        self._collector.collect('critic_loss', loss)
        return loss
