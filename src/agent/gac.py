from dataclasses import dataclass
from functools import partial
from typing import Any, List, NamedTuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import src.agent.components.networks.networks as nets
from src.agent.components.buffer import EnsembleReplayBuffer, TransitionBatch


class GACNetParams(NamedTuple):
    critic_params: chex.ArrayTree
    critic_target_params: chex.ArrayTree
    actor_params: chex.ArrayTree
    proposal_params: chex.ArrayTree

class GACOptimizerStates(NamedTuple):
    critic_opt_state: Any # TODO: Unsure about type hint
    actor_opt_state: Any
    proposal_opt_state: Any


class GACState(NamedTuple):
    params: GACNetParams
    opt_states: GACOptimizerStates


@dataclass
class GreedyACConfig:
    num_samples: int = 128
    actor_percentile: float = 0.1
    proposal_percentile: float = 0.2
    uniform_weight: float = 1.0
    batch_size: int = 256


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


def actor_builder(cfg: nets.TorsoConfig, act_dim: int):
    def _inner(x: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x)

        return ActorOutputs(
            alpha=hk.Linear(act_dim)(phi),
            beta=hk.Linear(act_dim)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int):
        self.seed = seed
        self.action_dim = action_dim
        self.num_samples = cfg.num_samples
        self.actor_percentile = cfg.actor_percentile
        self.proposal_percentile = cfg.proposal_percentile
        self.uniform_weight = cfg.uniform_weight
        self.batch_size = cfg.batch_size

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
        self.actor = actor_builder(actor_torso_cfg, action_dim)
        self.proposal = self.actor

        key = jax.random.PRNGKey(seed)
        rngs = jax.random.split(key, 4)
        params = GACNetParams(
            critic_params=self.critic.init(rng=rngs[0], x=jnp.zeros(state_dim), a=jnp.zeros(action_dim)),
            critic_target_params=self.critic.init(rng=rngs[1], x=jnp.zeros(state_dim), a=jnp.zeros(action_dim)),
            actor_params=self.actor.init(rng=rngs[2], x=jnp.zeros(state_dim)),
            proposal_params=self.proposal.init(rng=rngs[3], x=jnp.zeros(state_dim)),
        )

        # Optimizers
        critic_lr = 0.001
        self.critic_opt = optax.adam(critic_lr)
        critic_opt_state = self.critic_opt.init(params.critic_params)
        actor_lr = 0.001
        self.actor_opt = optax.adam(actor_lr)
        actor_opt_state = self.actor_opt.init(params.actor_params)
        proposal_lr = 0.001
        self.proposal_opt = optax.adam(proposal_lr)
        proposal_opt_state = self.proposal_opt.init(params.proposal_params)
        opt_states = GACOptimizerStates(
            critic_opt_state=critic_opt_state,
            actor_opt_state=actor_opt_state,
            proposal_opt_state=proposal_opt_state
        )

        # Replay Buffers
        self.critic_buffer = EnsembleReplayBuffer(
            n_ensemble=1,
            ensemble_prob=1.0
        )
        self.policy_buffer = EnsembleReplayBuffer(
            n_ensemble=1,
            ensemble_prob=1.0
        )

        self.agent_state = GACState(params, opt_states)


    @partial(jax.jit, static_argnums=(0,))
    def get_actor_actions(self, rng: chex.PRNGKey, states: jax.Array):
        actor_params = self.agent_state.params.actor_params
        out: ActorOutputs = self.actor.apply(params=actor_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)


    @partial(jax.jit, static_argnums=(0,))
    def get_proposal_actions(self, rng: chex.PRNGKey, states: jax.Array) -> jax.Array:
        proposal_params = self.agent_state.params.proposal_params
        out: ActorOutputs = self.proposal.apply(params=proposal_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    @partial(jax.jit, static_argnums=(0,))
    def get_uniform_actions(self, rng: chex.PRNGKey, samples: int) -> jax.Array:
        return jax.random.uniform(rng, (samples, self.action_dim))

    def critic_loss(self, batch: TransitionBatch):
        squared_errors = []
        # TODO: vmap this
        for transition in batch.steps:
            state = transition.prior.state
            action = transition.post.action
            reward = transition.n_step_reward
            next_state = transition.post.state
            gamma = transition.n_step_gamma
            # TODO: dp mask?

            next_action = self.actor.apply(params=self.agent_state.params.actor_params, x=next_state)
            target_value = self.critic.apply(
                params=self.agent_state.params.critic_target_params,
                x=[next_state, next_action]
            )
            target = reward + gamma * target_value
            value = self.critic.apply(params=self.agent_state.params.critic_params, x=[state, action])
            loss = jnp.square(target - value)
            squared_errors.append(loss)

        return jnp.mean(jnp.array(squared_errors))

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
        sampled_actions = self.get_action_samples(states)
        repeat_states = states.repeat(self.num_samples, axis=0)



    def actor_loss(self):
        pass

    def proposal_loss(self):
        pass

    def critic_update(self):
        batch = self.critic_buffer.sample(self.batch_size)
        loss, grads = jax.value_and_grad(self.critic_loss)(self.agent_state.params.critic_params, batch)
        updates, updated_critic_opt_state = self.critic_opt.update(grads, self.agent_state.opt_states.critic_opt_state)
        self.agent_state.opt_states.critic_opt_state = updated_critic_opt_state
        updated_critic_params = optax.apply_updates(self.agent_state.params.critic_params, updates)
        self.agent_state.params.critic_params = updated_critic_params

        # Target Net Polyak Update
        polyak = 0.995
        target_params = self.agent_state.params.critic_target_params
        updated_target_params = optax.incremental_update(updated_critic_params, target_params, polyak)
        self.agent_state.params.critic_target_params = updated_target_params

    def actor_update(self, params: GACNetParams):
        batch = self.policy_buffer.sample(self.batch_size)
        states = batch[0].steps # TODO: wrong


    def proposal_update(self, params: GACNetParams):
        pass

    def update(self, params: GACNetParams):
        pass
