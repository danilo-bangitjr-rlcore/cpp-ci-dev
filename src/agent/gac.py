from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import src.agent.components.networks.networks as nets
from src.agent.components.buffer import EnsembleReplayBuffer, TransitionBatch
from src.agent.components.networks.activations import ActivationConfig, TanhConfig, get_output_activation, scale_shift
from src.interaction.transition_creator import Transition


class CriticState(NamedTuple):
    params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: Any


class PolicyState(NamedTuple):
    actor_params: chex.ArrayTree
    proposal_params: chex.ArrayTree
    actor_opt_state: Any
    proposal_opt_state: Any


@dataclass
class GACState:
    critic: CriticState
    policy: PolicyState


@dataclass
class GreedyACConfig:
    num_samples: int = 128
    actor_percentile: float = 0.1
    proposal_percentile: float = 0.2
    uniform_weight: float = 1.0
    batch_size: int = 256
    ensemble: int = 1


class CriticOutputs(NamedTuple):
    q: jax.Array


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        return CriticOutputs(
            q=hk.Linear(1)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class ActorOutputs(NamedTuple):
    alpha: jax.Array
    beta: jax.Array


def actor_builder(cfg: nets.TorsoConfig, act_cfg: ActivationConfig, act_dim: int):
    def _inner(x: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x)
        output_act = get_output_activation(act_cfg)
        alpha_head_out = output_act(hk.Linear(act_dim)(phi))
        beta_head_out = output_act(hk.Linear(act_dim)(phi))

        return ActorOutputs(
            alpha=scale_shift(alpha_head_out, 1, 10000),
            beta=scale_shift(beta_head_out, 1, 10000),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_samples = cfg.num_samples
        self.actor_percentile = cfg.actor_percentile
        self.proposal_percentile = cfg.proposal_percentile
        self.uniform_weight = cfg.uniform_weight
        self.ensemble = cfg.ensemble

        # Neural Nets
        critic_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(sizes=[32, 32], activation='relu'),
                nets.LinearConfig(size=64, activation='relu'),
            ],
        )
        self.critic = critic_builder(critic_torso_cfg)

        actor_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=64, activation='relu'),
                nets.LinearConfig(size=64, activation='relu'),
            ]
        )
        actor_output_act_cfg = TanhConfig(shift=-4.0)
        self.actor = actor_builder(actor_torso_cfg, actor_output_act_cfg, action_dim)
        self.proposal = self.actor

        # Optimizers
        critic_lr = 0.001
        self.critic_opt = optax.adam(critic_lr)
        actor_lr = 0.001
        self.actor_opt = optax.adam(actor_lr)
        proposal_lr = 0.001
        self.proposal_opt = optax.adam(proposal_lr)

        # Replay Buffers
        self.critic_buffer = EnsembleReplayBuffer(
            n_ensemble=self.ensemble,
            ensemble_prob=0.5,
            batch_size=cfg.batch_size,
        )
        self.policy_buffer = EnsembleReplayBuffer(
            n_ensemble=1,
            ensemble_prob=1.0,
            batch_size=cfg.batch_size,
        )

        # Agent State
        dummy_x = jnp.zeros(self.state_dim)
        dummy_a = jnp.zeros(self.action_dim)

        rngs = jax.random.split(self.rng, self.ensemble)
        critic_state = jax.vmap(self.get_critic_state, in_axes=(0, None, None))(rngs, dummy_x, dummy_a)

        self.rng, rng = jax.random.split(self.rng)
        actor_params = self.actor.init(rng=self.rng, x=dummy_x)
        self.rng, rng = jax.random.split(self.rng)
        proposal_params = self.proposal.init(rng=self.rng, x=dummy_x)

        actor_opt_state = self.actor_opt.init(actor_params)
        proposal_opt_state = self.proposal_opt.init(proposal_params)

        policy_state = PolicyState(
            actor_params=actor_params,
            proposal_params=proposal_params,
            actor_opt_state=actor_opt_state,
            proposal_opt_state=proposal_opt_state
        )

        self.agent_state = GACState(critic_state, policy_state)

    @partial(jax.jit, static_argnums=(0,))
    def get_critic_state(self, rng: chex.PRNGKey, dummy_x: jax.Array, dummy_a: jax.Array):
        # Critic
        rng, _ = jax.random.split(rng)
        critic_params = self.critic.init(rng, dummy_x, dummy_a)

        # Target
        rng, _ = jax.random.split(rng)
        target_params = self.critic.init(rng, dummy_x, dummy_a)

        # Optimizer
        critic_opt_state = self.critic_opt.init(critic_params)

        return CriticState(critic_params, target_params, critic_opt_state)

    def update_buffer(self, transition: Transition):
        self.critic_buffer.add(transition)
        self.policy_buffer.add(transition)

    @partial(jax.jit, static_argnums=(0,))
    def get_actor_actions(self, rng: chex.PRNGKey, states: jax.Array):
        actor_params = self.agent_state.policy.actor_params
        out: ActorOutputs = self.actor.apply(params=actor_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    @partial(jax.jit, static_argnums=(0,))
    def get_actions(self, rng: chex.PRNGKey, states: jax.Array):
        return self.get_actor_actions(rng, states)

    @partial(jax.jit, static_argnums=(0,))
    def get_proposal_actions(self, rng: chex.PRNGKey, states: jax.Array) -> jax.Array:
        proposal_params = self.agent_state.policy.proposal_params
        out: ActorOutputs = self.proposal.apply(params=proposal_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    @partial(jax.jit, static_argnums=(0,))
    def get_uniform_actions(self, rng: chex.PRNGKey, samples: int) -> jax.Array:
        return jax.random.uniform(rng, (samples, self.action_dim))

    def critic_loss(
            self,
            critic_params: chex.ArrayTree,
            target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            transition: Transition,
            rng: chex.PRNGKey
    ):
        prior = transition.steps[0]
        post = transition.steps[-1]

        state = prior.state
        action = post.action
        reward = transition.n_step_reward
        next_state = post.state
        gamma = transition.n_step_gamma

        actor_output = self.actor.apply(params=actor_params, x=next_state)
        next_action = distrax.Beta(actor_output.alpha, actor_output.beta).sample(seed=rng)

        target_value = self.critic.apply(params=target_params, x=next_state, a=next_action)
        target = reward + gamma * target_value.q
        value = self.critic.apply(params=critic_params, x=state, a=action)
        loss = jnp.square(target - value.q)

        return loss

    def _batch_critic_loss(
            self,
            critic_params: chex.ArrayTree,
            target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            batch: TransitionBatch,
            rng: chex.PRNGKey
    ):
        rngs = jax.random.split(rng, len(batch.steps))
        losses = jax.vmap(self.critic_loss, in_axes=(None, None, None, 0, 0))(critic_params,
                                                                              target_params,
                                                                              actor_params,
                                                                              batch.steps,
                                                                              rngs)

        return jnp.mean(losses)

    def _critic_update(
            self,
            critic_state: CriticState,
            actor_params: chex.ArrayTree,
            batch: TransitionBatch,
            rng: chex.PRNGKey
    ):
        loss, grads = jax.value_and_grad(self._batch_critic_loss)(critic_state.params,
                                                                  critic_state.target_params,
                                                                  actor_params,
                                                                  batch,
                                                                  rng)
        updates, updated_critic_opt_state = self.critic_opt.update(grads, critic_state.opt_state)
        updated_critic_params = optax.apply_updates(critic_state.params, updates)

        # Target Net Polyak Update
        polyak = 0.995
        target_params = critic_state.target_params
        updated_target_params = optax.incremental_update(updated_critic_params, target_params, polyak)

        return CriticState(
            updated_critic_params,
            updated_target_params,
            updated_critic_opt_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _ensemble_critic_update(self, batches: list[TransitionBatch]):
        rngs = jax.random.split(self.rng, self.ensemble)
        new_critic_state = jax.vmap(self._critic_update, in_axes=(0, None, 0, 0))(self.agent_state.critic,
                                                                                  self.agent_state.policy.actor_params,
                                                                                  batches,
                                                                                  rngs)

        return new_critic_state

    def critic_update(self):
        batches = self.critic_buffer.sample(self.batch_size)
        new_critic_state = self._ensemble_critic_update(batches)
        self.agent_state.critic = new_critic_state

    def get_action_samples(self, states: jax.Array):
        uniform_samples = int(self.num_samples * self.uniform_weight)
        proposal_samples = self.num_samples - uniform_samples
        num_states = len(states)

        key = jax.random.PRNGKey(self.seed)
        uniform_actions = self.get_uniform_actions(key, num_states * uniform_samples)
        uniform_actions = uniform_actions.reshape((num_states, uniform_samples, self.action_dim))

        repeat_states = states.repeat(proposal_samples, axis=0)
        proposal_actions = self.get_proposal_actions(key, repeat_states)
        proposal_actions.reshape((num_states, proposal_samples, self.action_dim))
        sampled_actions = jnp.concat([uniform_actions, proposal_actions], axis=1)

        return sampled_actions

    def get_top_percentile_actions(self, states: jax.Array):
        self.get_action_samples(states)
        states.repeat(self.num_samples, axis=0)

    def actor_loss(self):
        pass

    def proposal_loss(self):
        pass

    def actor_update(self):
        if self.policy_buffer.size == 0:
            return
        self.policy_buffer.sample()

    def proposal_update(self):
        pass

    def update(self):
        self.critic_update()
        self.actor_update()
        self.proposal_update()
