from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.actor_registry import get_actor
from lib_agent.actor.percentile_actor import PAState, State
from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.critic.adv_critic import AdvCritic
from lib_agent.critic.critic_registry import get_critic
from lib_agent.critic.critic_utils import CriticState
from lib_agent.critic.qrc_critic import QRCCritic
from lib_utils.named_array import NamedArray
from ml_instrumentation.Collector import Collector

from agent.interface import Batch
from interaction.transition_creator import Transition


class GACState(NamedTuple):
    critic: CriticState
    actor: PAState


@dataclass
class GreedyACConfig:
    name: str
    batch_size: int
    critic: dict[str, Any]
    actor: dict[str, Any]


class BatchWithState(NamedTuple):
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._cfg = cfg
        self._collector = collector

        self._critic = get_critic(cfg.critic, seed, state_dim, action_dim)
        self._actor = get_actor(cfg.actor, seed, state_dim, action_dim)

        # Replay Buffers
        self.policy_buffer = EnsembleReplayBuffer[Batch](
            ensemble=1,
            ensemble_probability=1.0,
            batch_size=cfg.batch_size,
        )

        self.critic_buffer = EnsembleReplayBuffer[Batch](
            ensemble=cfg.critic['ensemble'],
            ensemble_probability=cfg.critic['ensemble_prob'],
            batch_size=cfg.batch_size,
        )

        # Agent State
        dummy_x = jnp.zeros(self.state_dim)
        dummy_a = jnp.zeros(self.action_dim)

        self.rng, c_rng = jax.random.split(self.rng)
        critic_state = self._critic.init_state(c_rng, dummy_x, dummy_a)
        actor_state = self._actor.init_state(c_rng, dummy_x)

        self.agent_state = GACState(critic_state, actor_state)

    def update_buffer(self, transition: Transition):
        t = Batch(
            state=NamedArray.unnamed(jnp.asarray(transition.state)),
            action=jnp.array(transition.action),
            reward=jnp.array([transition.reward]),
            next_state=NamedArray.unnamed(jnp.array(transition.next_state)),
            gamma=jnp.array([transition.gamma]),

            a_lo=jnp.array(transition.a_lo),
            a_hi=jnp.array(transition.a_hi),
            next_a_lo=jnp.array(transition.next_a_lo),
            next_a_hi=jnp.array(transition.next_a_hi),

            last_a=jnp.array(transition.last_a),
            dp=jnp.array(transition.dp),
            next_dp=jnp.array(transition.next_dp),
        )
        self.critic_buffer.add(t)
        self.policy_buffer.add(t)

    def get_actions(self, state: State, n_samples: int = 1):
        actions, _ = self._actor.get_actions(self.agent_state.actor.actor.params, state, n_samples)
        chex.assert_shape(actions, (n_samples, self.action_dim))
        if n_samples == 1:
            return actions.squeeze(0)
        return actions

    def get_action_values(self, state: State, actions: jax.Array | np.ndarray):
        self.rng, c_rng = jax.random.split(self.rng)
        # use get_active_values instead of vmapping over all critics
        return self._critic.get_active_values(
            self.agent_state.critic.params,
            c_rng,
            state=state.features.array,
            action=jnp.asarray(actions),
        ).q

    def get_probs(self, actor_params: chex.ArrayTree, state: State, actions: jax.Array | np.ndarray):
        actions = jnp.asarray(actions)
        return self._actor.get_probs(actor_params, state, actions)

    def update(self):
        self.critic_update()
        self.policy_update()

    def critic_update(self):
        assert isinstance(self._critic, QRCCritic)
        if self.critic_buffer.size == 0:
            return

        batch = self.critic_buffer.sample()
        next_state = State(
            features=batch.next_state,
            a_lo=batch.next_a_lo,
            a_hi=batch.next_a_hi,
            last_a=batch.action,
            dp=jnp.expand_dims(batch.next_dp, axis=-1),
        )
        self.rng, bs_rng = jax.random.split(self.rng)
        next_actions, _ = self._actor.get_actions_rng(
            self.agent_state.actor.actor.params,
            bs_rng,
            next_state,
            10,
        )
        new_critic_state, metrics = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=BatchWithState(
                state=State(
                    features=batch.state,
                    a_lo=batch.a_lo,
                    a_hi=batch.a_hi,
                    last_a=batch.last_a,
                    dp=batch.dp,
                ),
                action=batch.action,
                reward=batch.reward.squeeze(-1),
                next_state=next_state,
                gamma=batch.gamma.squeeze(-1),
            ),
            next_actions=next_actions,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)
        self._collector.collect('critic_loss', metrics.loss.mean().item())

    def initialize_to_nominal_action(self, nominal_setpoints: jax.Array, iterations: int = 100):
        critic_rng, actor_rng, self.rng = jax.random.split(self.rng, 3)
        new_critic_state = self._critic.initialize_to_nominal_action(
            critic_rng,
            self.agent_state.critic,
            nominal_setpoints,
        )
        self.agent_state = self.agent_state._replace(critic=new_critic_state)

        new_actor_state = self._actor.initialize_to_nominal_action(
            actor_rng,
            self.agent_state.actor.actor,
            nominal_setpoints,
            self.state_dim,
        )
        new_actor_state = PAState(new_actor_state, self.agent_state.actor.proposal)

        self.agent_state = self.agent_state._replace(actor=new_actor_state)

    def policy_update(self):
        if self.policy_buffer.size == 0:
            return

        batch = self.policy_buffer.sample()

        actor_state, metrics = self._actor.update(
            self.agent_state.actor,
            self.ensemble_ve,
            self.agent_state.critic.params,
            BatchWithState(
                state=State(
                    features=batch.state,
                    a_lo=batch.a_lo,
                    a_hi=batch.a_hi,
                    last_a=batch.last_a,
                    dp=jnp.expand_dims(batch.dp, axis=-1),
                ),
                action=batch.action,
                reward=batch.reward,
                next_state=State(
                    features=batch.next_state,
                    a_lo=batch.next_a_lo,
                    a_hi=batch.next_a_hi,
                    last_a=batch.action,
                    dp=jnp.expand_dims(batch.next_dp, axis=-1),
                ),
                gamma=batch.gamma,
            ),
        )
        self.agent_state = self.agent_state._replace(actor=actor_state)

        self._collector.collect('actor_loss', metrics.actor_loss.mean().item())

    def ensemble_ve(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        qs = self._critic.get_active_values(params, rng, x, a).q
        values = qs.mean(axis=0).squeeze(-1)

        chex.assert_rank(values, 0)
        return values


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
        chex.assert_shape(policy_probs, (ensemble_size, batch_size, num_policy_actions))
        return policy_actions, policy_probs

    def critic_update(self):
        assert isinstance(self._critic, AdvCritic)
        if self.critic_buffer.size == 0:
            return

        batch = self.critic_buffer.sample()

        # Create state for current timestep
        states = State(
            features=batch.state,
            a_lo=batch.a_lo,
            a_hi=batch.a_hi,
            last_a=batch.last_a,
            dp=jnp.expand_dims(batch.next_dp, axis=-1),
        )

        policy_actions, policy_probs = self._get_actions_and_probs(states)

        # Create next state
        next_states = State(
            features=batch.next_state,
            a_lo=batch.next_a_lo,
            a_hi=batch.next_a_hi,
            last_a=batch.action,
            dp=jnp.expand_dims(batch.next_dp, axis=-1),
        )

        # Update critic with policy actions and probabilities
        new_critic_state, metrics = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=BatchWithState(
                state=states,
                action=batch.action,
                reward=batch.reward.squeeze(-1),
                next_state=next_states,
                gamma=batch.gamma.squeeze(-1),
            ),
            policy_actions=policy_actions,
            policy_probs=policy_probs,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)
        self._collector.collect('critic_loss', metrics.loss.mean().item())
