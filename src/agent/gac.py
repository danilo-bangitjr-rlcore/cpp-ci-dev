from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, cast

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import src.agent.components.networks.networks as nets
from src.agent.components.buffer import EnsembleReplayBuffer, VectorizedTransition
from src.agent.components.networks.activations import ActivationConfig, TanhConfig, get_output_activation, scale_shift
from src.interaction.transition_creator import Transition


class CriticState(NamedTuple):
    params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: Any


class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: Any


class GACState(NamedTuple):
    critic: CriticState
    actor: PolicyState
    proposal: PolicyState


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
        critic_state = jax.vmap(self.init_critic_state, in_axes=(0, None, None))(rngs, dummy_x, dummy_a)

        self.rng, rng = jax.random.split(self.rng)
        actor_params = self.actor.init(rng=rng, x=dummy_x)
        actor_state = PolicyState(
            params=actor_params,
            opt_state=self.actor_opt.init(actor_params),
        )

        self.rng, rng = jax.random.split(self.rng)
        proposal_params = self.proposal.init(rng=rng, x=dummy_x)
        proposal_state = PolicyState(
            params=proposal_params,
            opt_state=self.proposal_opt.init(proposal_params),
        )

        self.agent_state = GACState(critic_state, actor_state, proposal_state)

    @partial(jax.jit, static_argnums=(0,))
    def init_critic_state(self, rng: chex.PRNGKey, dummy_x: jax.Array, dummy_a: jax.Array):
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
    def _get_actions(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array):
        out: ActorOutputs = self.actor.apply(params=params, x=state)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    def get_actions(self, state: jax.Array):
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        return self._get_actions(self.agent_state.actor.params, sample_rng, state )

    @partial(jax.jit, static_argnums=(0,))
    def get_uniform_actions_asdf(self, rng: chex.PRNGKey, samples: int) -> jax.Array:
        return jax.random.uniform(rng, (samples, self.action_dim))


    # ---------------------------------------------------------------------------- #
    #                                Critic Updating                               #
    # ---------------------------------------------------------------------------- #

    def critic_loss(
            self,
            rng: chex.PRNGKey,
            critic_params: chex.ArrayTree,
            target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            transition: VectorizedTransition,
    ):
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma

        next_action = self.get_actions(actor_params, rng, next_state)
        target_value = self.critic.apply(params=target_params, x=next_state, a=next_action)
        target = reward + gamma * target_value.q
        value = self.critic.apply(params=critic_params, x=state, a=action)
        loss = jnp.square(target - value.q)

        return loss

    def _batch_critic_loss(
            self,
            rng: chex.PRNGKey,
            critic_params: chex.ArrayTree,
            target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            transitions: VectorizedTransition,
    ):
        rngs = jax.random.split(rng, self.critic_buffer.batch_size)
        vmapped = jax.vmap(self.critic_loss, in_axes=(0, None, None, None, 0))
        losses = vmapped(
            rngs,
            critic_params,
            target_params,
            actor_params,
            transitions,
        )

        return jnp.mean(losses)

    def _member_critic_update(
            self,
            rng: chex.PRNGKey,
            critic_state: CriticState,
            actor_state: PolicyState,
            transitions: VectorizedTransition,
    ) -> CriticState:
        """
        Updates a single member of the ensemble.
        """
        grad_fn = jax.grad(self._batch_critic_loss)
        grads = grad_fn(
            rng,
            critic_state.params,
            critic_state.target_params,
            actor_state.params,
            transitions,
        )
        updates, new_opt_state = self.critic_opt.update(grads, critic_state.opt_state)
        new_params = optax.apply_updates(critic_state.params, updates)

        # Target Net Polyak Update
        polyak = 0.995
        target_params = critic_state.target_params
        new_target_params = optax.incremental_update(new_params, target_params, polyak)

        return CriticState(
            new_params,
            new_target_params,
            new_opt_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _ensemble_critic_update(
        self,
        rng: chex.PRNGKey,
        critic_state: CriticState,
        actor_state: PolicyState,
        transitions: VectorizedTransition,
    ) -> CriticState:
        """
        Updates each member of the ensemble.
        """
        rngs = jax.random.split(rng, self.ensemble)
        vmapped = jax.vmap(self._member_critic_update, in_axes=(0, 0, None, 0))
        new_critic_state =  vmapped(
            rngs,
            critic_state,
            actor_state,
            transitions,
        )
        return new_critic_state

    def critic_update(self):
        if self.critic_buffer.size == 0:
            return
        transitions = self.critic_buffer.sample()

        self.rng, update_key = jax.random.split(self.rng, 2)
        new_critic_state = self._ensemble_critic_update(
            update_key,
            self.agent_state.critic,
            self.agent_state.actor,
            transitions,
        )
        self.agent_state = self.agent_state._replace(critic_state=new_critic_state)


    def get_action_samples(self, state: jax.Array):
        uniform_samples = int(self.num_samples * self.uniform_weight)
        proposal_samples = self.num_samples - uniform_samples

        self.rng, rng = jax.random.split(self.rng)
        uniform_actions = self.get_uniform_actions(self.rng, uniform_samples)

        rngs = jax.random.split(self.rng, proposal_samples)
        proposal_actions = jax.vmap(self.get_proposal_action, in_axes=(0, None))(rngs, state)

        sampled_actions = jnp.concat([uniform_actions, proposal_actions], axis=1)

        return sampled_actions

    def ensemble_state_action_eval(self, state: jax.Array, action: jax.Array):
        q_vals = jax.vmap(self.critic.apply, in_axes=(0, None, None))(self.agent_state.critic.params, state, action)
        q_val = jnp.mean(q_vals) # Mean Reduction
        return q_val

    def get_top_percentile_actions(self, state: jax.Array):
        sampled_actions = self.get_action_samples(state)
        q_vals = jax.vmap(self.ensemble_state_action_eval, in_axes=(None, 0))(state, sampled_actions)
        sorted_inds = jnp.argsort(q_vals, descending=True, axis=None)

        # Actor
        actor_top_n = int(self.actor_percentile * self.num_samples)
        actor_top_n_inds = sorted_inds[:actor_top_n]
        actor_update_actions = sampled_actions[actor_top_n_inds]

        proposal_top_n = int(self.proposal_percentile * self.num_samples)
        proposal_top_n_inds = sorted_inds[:proposal_top_n]
        proposal_update_actions = sampled_actions[proposal_top_n_inds]

        return actor_update_actions, proposal_update_actions


    def actor_loss(self, state: jax.Array, top_actions: jax.Array):
        actor_params = self.agent_state.policy.actor_params
        out: ActorOutputs = self.actor.apply(params=actor_params, x=state)
        dist = distrax.Beta(out.alpha, out.beta)
        log_prob = cast(jax.Array, dist.log_prob(top_actions))

        return -log_prob

    def batch_actor_loss(self, states: jax.Array, top_actions_batch: jax.Array):
        losses = jax.vmap(self.actor_loss, in_axes=(0,0))(states, top_actions_batch)

        # TODO: axis?
        return jnp.mean(losses)

    def proposal_loss(self, state: jax.Array, top_actions: jax.Array):
        proposal_params = self.agent_state.policy.proposal_params
        out: ActorOutputs = self.proposal.apply(params=proposal_params, x=state)
        dist = distrax.Beta(out.alpha, out.beta)
        log_prob = cast(jax.Array, dist.log_prob(top_actions))

        return -log_prob

    def batch_proposal_loss(self, states: jax.Array, top_actions_batch: jax.Array):
        losses = jax.vmap(self.proposal_loss, in_axes=(0, 0))(states, top_actions_batch)

        # TODO: axis?
        return jnp.mean(losses)

    def _actor_update(self, policy_state: PolicyState, states: jax.Array, update_actions: jax.Array):
        grads = jax.grad(self.batch_actor_loss)(states, update_actions)
        updates, new_opt_state = self.actor_opt.update(grads, policy_state.actor_opt_state)
        new_params = optax.apply_updates(policy_state.actor_params, updates)

        new_state = policy_state._replace(actor_params=new_params, actor_opt_state=new_opt_state)
        return new_state

    def _proposal_update(self, policy_state: PolicyState, states: jax.Array, update_actions: jax.Array):
        grads = jax.grad(self.batch_proposal_loss)(states, update_actions)
        updates, new_opt_state = self.proposal_opt.update(grads, policy_state.proposal_opt_state)
        new_params = optax.apply_updates(policy_state.proposal_params, updates)

        new_policy_state = policy_state._replace(proposal_params=new_params, proposal_opt_state=new_opt_state)
        return new_policy_state

    def _policy_update(self, states: jax.Array):
        actor_update_actions, proposal_update_actions = jax.vmap(self.get_top_percentile_actions, in_axes=(0))(states)

        # Actor Update
        new_actor_state = self._actor_update(self.agent_state.policy, states, actor_update_actions)
        self.agent_state = self.agent_state._replace(policy=new_actor_state)

        # Proposal Update
        updated_proposal_params = self._proposal_update(self.agent_state.policy, states, proposal_update_actions)
        self.agent_state.policy.proposal_params = updated_proposal_params

    def policy_update(self):
        if self.policy_buffer.size == 0:
            return
        states = self.policy_buffer.sample()
        self._policy_update(states)

    def update(self):
        self.critic_update()
        self.policy_update()
