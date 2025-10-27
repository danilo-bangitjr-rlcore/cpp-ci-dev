from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.actor_registry import get_actor
from lib_agent.actor.percentile_actor import PAState
from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.buffer.datatypes import State, Transition
from lib_agent.critic.adv_critic import AdvCritic
from lib_agent.critic.critic_registry import get_critic
from lib_agent.critic.critic_utils import CriticState
from lib_agent.critic.qrc_critic import QRCCritic
from ml_instrumentation.Collector import Collector


def exp_moving_avg(decay: float, mean: float | None, x: float) -> float:
    """Computes exponential moving average: mean = decay * mean + (1 - decay) * x"""
    if mean is None:
        return x
    return decay * mean + (1 - decay) * x


class GACState(NamedTuple):
    critic: CriticState
    actor: PAState


@dataclass
class GreedyACConfig:
    name: str
    batch_size: int
    critic: dict[str, Any]
    actor: dict[str, Any]
    max_critic_updates: int = 10
    max_internal_actor_updates: int = 3
    loss_ema_factor: float = 0.75
    loss_threshold: float = 1e-4
    bootstrap_action_samples: int = 10


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
        self.policy_buffer = EnsembleReplayBuffer[Transition](
            ensemble=1,
            ensemble_probability=1.0,
            batch_size=cfg.batch_size,
        )

        self.critic_buffer = EnsembleReplayBuffer[Transition](
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

        # for early stopping
        self._last_critic_loss = 0.
        self._avg_critic_delta: float | None = None
        self._last_actor_loss = 0.
        self._avg_actor_delta: float | None = None

    def update_buffer(self, transition: Transition):
        self.critic_buffer.add(transition)
        self.policy_buffer.add(transition)

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
        alpha = self._cfg.loss_ema_factor

        for _ in range(self._cfg.max_critic_updates):
            critic_loss = self.critic_update()

            for _ in range(self._cfg.max_internal_actor_updates):
                actor_loss = self.policy_update()
                last = self._last_actor_loss
                self._last_actor_loss = actor_loss
                delta = actor_loss - last
                self._avg_actor_delta = exp_moving_avg(alpha, self._avg_actor_delta, delta)

                if np.abs(self._avg_actor_delta) < self._cfg.loss_threshold:
                    break

            last = self._last_critic_loss
            self._last_critic_loss = critic_loss
            delta = critic_loss - last
            self._avg_critic_delta = exp_moving_avg(alpha, self._avg_critic_delta, delta)

            if np.abs(self._avg_critic_delta) < self._cfg.loss_threshold:
                break

    def critic_update(self):
        assert isinstance(self._critic, QRCCritic)
        if self.critic_buffer.size == 0:
            return 0.

        transitions: Transition = self.critic_buffer.sample()
        self.rng, bs_rng = jax.random.split(self.rng)
        next_actions, _ = self._actor.get_actions_rng(
            self.agent_state.actor.actor.params,
            bs_rng,
            transitions.next_state,
            self._cfg.bootstrap_action_samples,
        )
        new_critic_state, metrics = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=transitions,
            next_actions=next_actions,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)
        loss = metrics.loss.mean().item()
        self._collector.collect('critic_loss', loss)
        return loss

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
            return 0.

        transitions: Transition = self.policy_buffer.sample()

        actor_state, metrics = self._actor.update(
            self.agent_state.actor,
            self.ensemble_ve,
            self.agent_state.critic.params,
            transitions,
        )
        self.agent_state = self.agent_state._replace(actor=actor_state)

        loss = metrics.actor_loss.mean().item()
        self._collector.collect('actor_loss', loss)
        return loss

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
            return 0.

        transitions: Transition = self.critic_buffer.sample()

        policy_actions, policy_probs = self._get_actions_and_probs(transitions.state)

        # Update critic with policy actions and probabilities
        new_critic_state, metrics = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=transitions,
            policy_actions=policy_actions,
            policy_probs=policy_probs,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)
        loss = metrics.loss.mean().item()
        self._collector.collect('critic_loss', loss)
        return loss
