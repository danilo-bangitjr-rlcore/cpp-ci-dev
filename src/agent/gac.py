from dataclasses import dataclass
from typing import NamedTuple, cast

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_instrumentation.Collector import Collector

import agent.components.networks.networks as nets
import utils.jax as jax_u
from agent.components.buffer import EnsembleReplayBuffer, VectorizedTransition
from agent.components.networks.activations import ActivationConfig, TanhConfig, get_output_activation, scale_shift
from interaction.transition_creator import Transition


class UpdateActions(NamedTuple):
    actor: jax.Array
    proposal: jax.Array


class CriticState(NamedTuple):
    params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: chex.ArrayTree


class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree


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
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_samples = cfg.num_samples
        self.actor_percentile = cfg.actor_percentile
        self.proposal_percentile = cfg.proposal_percentile
        self.uniform_weight = cfg.uniform_weight
        self.ensemble = cfg.ensemble

        self._collector = collector

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
        self.critic_opt = optax.chain(
            optax.adam(learning_rate=critic_lr),
            # optax.scale_by_backtracking_linesearch(
            #     max_backtracking_steps=50, max_learning_rate=critic_lr, decrease_factor=0.9, slope_rtol=0.1
            # ),
        )

        actor_lr = 0.001
        self.actor_opt = optax.adam(actor_lr)
        proposal_lr = 0.001
        self.proposal_opt = optax.adam(proposal_lr)

        # Replay Buffers
        self.critic_buffer = EnsembleReplayBuffer(
            n_ensemble=self.ensemble,
            ensemble_prob=1.0,
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

    @jax_u.method_jit
    def init_critic_state(self, rng: chex.PRNGKey, dummy_x: jax.Array, dummy_a: jax.Array):
        # Critic
        rng, _ = jax.random.split(rng)
        critic_params = self.critic.init(rng, dummy_x, dummy_a)

        # Target
        rng, _ = jax.random.split(rng)
        target_params = critic_params

        # Optimizer
        critic_opt_state = self.critic_opt.init(critic_params)

        return CriticState(critic_params, target_params, critic_opt_state)

    def update_buffer(self, transition: Transition):
        self.critic_buffer.add(transition)
        self.policy_buffer.add(transition)

    @jax_u.method_jit
    def _get_actions(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array):
        out: ActorOutputs = self.actor.apply(params=params, x=state)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    def get_actions(self, state: jax.Array | np.ndarray):

        state = jnp.asarray(state)
        if state.ndim == 1:
            state = jnp.expand_dims(state, axis=0)
            self.rng, sample_rng = jax.random.split(self.rng, 2)
            return self._get_actions(self.agent_state.actor.params, sample_rng, state)[0]
        else:
            self.rng, sample_rng = jax.random.split(self.rng, 2)
            return self._get_actions(self.agent_state.actor.params, sample_rng, state)

    def get_uniform_actions(self, rng: chex.PRNGKey, samples: int) -> jax.Array:
        return jax.random.uniform(rng, (samples, self.action_dim))


    # ---------------------------------------------------------------------------- #
    #                                Critic Updating                               #
    # ---------------------------------------------------------------------------- #

    def _get_target_q(self, target_params: chex.ArrayTree, state: jax.Array, action: jax.Array):
        return self.critic.apply(params=target_params, x=state, a=action).q

    def critic_loss(
            self,
            critic_params: chex.ArrayTree,
            ens_target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            transition: VectorizedTransition,
            rng: chex.PRNGKey,
    ):
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma

        next_action = self._get_actions(actor_params, rng, next_state)
        ens_targets = jax.vmap(self._get_target_q, in_axes=(0, None, None))(ens_target_params, next_state, next_action)
        target_value = ens_targets.mean(axis=0)
        target = reward + gamma * target_value
        value = self.critic.apply(params=critic_params, x=state, a=action)
        loss = jnp.square(target - value.q)

        return loss

    def _batch_critic_loss(
            self,
            critic_params: chex.ArrayTree,
            ens_target_params: chex.ArrayTree,
            actor_params: chex.ArrayTree,
            transitions: VectorizedTransition,
            rng: chex.PRNGKey,
    ):
        rngs = jax.random.split(rng, self.critic_buffer.batch_size)
        vmapped = jax.vmap(self.critic_loss, in_axes=(None, None, None, 0, 0))
        losses = vmapped(
            critic_params,
            ens_target_params,
            actor_params,
            transitions,
            rngs,
        )

        return jnp.mean(losses)

    def _member_critic_update(
            self,
            critic_state: CriticState,
            ens_target_params: chex.ArrayTree,
            actor_state: PolicyState,
            transitions: VectorizedTransition,
            rng: chex.PRNGKey,
    ):
        """
        Updates a single member of the ensemble.
        """
        loss, grads = jax.value_and_grad(self._batch_critic_loss)(
            critic_state.params,
            ens_target_params,
            actor_state.params,
            transitions,
            rng,
        )
        updates, new_opt_state = self.critic_opt.update(
            grads,
            critic_state.opt_state,
            critic_state.params,
            value=loss,
            grad=grads,
            value_fn=self._batch_critic_loss,
            ens_target_params=ens_target_params,
            actor_params=actor_state.params,
            transitions=transitions,
            rng=rng,
        )
        new_params = optax.apply_updates(critic_state.params, updates)

        # Target Net Polyak Update
        polyak = 0.995
        target_params = critic_state.target_params
        new_target_params = optax.incremental_update(new_params, target_params, polyak)

        return CriticState(
            new_params,
            new_target_params,
            new_opt_state,
        ), loss

    @jax_u.method_jit
    def _ensemble_critic_update(
        self,
        critic_state: CriticState,
        actor_state: PolicyState,
        transitions: VectorizedTransition,
        rng: chex.PRNGKey,
    ):
        """
        Updates each member of the ensemble.
        """
        rngs = jax.random.split(rng, self.ensemble)
        vmapped = jax.vmap(self._member_critic_update, in_axes=(0, None, None, 0, 0))
        new_critic_state, losses =  vmapped(
            critic_state,
            critic_state.target_params,
            actor_state,
            transitions,
            rngs,
        )
        return new_critic_state, losses

    def critic_update(self):
        if self.critic_buffer.size == 0:
            return
        transitions = self.critic_buffer.sample()

        self.rng, update_rng = jax.random.split(self.rng, 2)
        new_critic_state, losses = self._ensemble_critic_update(
            self.agent_state.critic,
            self.agent_state.actor,
            transitions,
            update_rng,
        )
        self.agent_state = self.agent_state._replace(critic=new_critic_state)

        loss = jnp.mean(losses)
        self._collector.collect('critic_loss', float(loss))


    def _get_action_samples(self, proposal_params: chex.ArrayTree, state: jax.Array, rng: chex.PRNGKey):
        uniform_samples = int(self.num_samples * self.uniform_weight)
        uniform_actions = self.get_uniform_actions(rng, uniform_samples)

        proposal_samples = self.num_samples - uniform_samples
        if proposal_samples == 0:
            return uniform_actions

        rngs = jax.random.split(self.rng, proposal_samples)
        proposal_actions = jax.vmap(self._get_actions, in_axes=(None, 0, None))(proposal_params, rngs, state)

        sampled_actions = jnp.concat([uniform_actions, proposal_actions], axis=0)

        return sampled_actions

    def _ensemble_state_action_eval(self, critic_params: chex.ArrayTree, state: jax.Array, action: jax.Array):
        q_vals = jax.vmap(self.critic.apply, in_axes=(0, None, None))(critic_params, state, action)
        q_val = jnp.mean(q_vals.q, axis=0) # Mean Reduction

        return q_val

    def _evaluate_sampled_actions(
        self,
        critic_params: chex.ArrayTree,
        proposal_params: chex.ArrayTree,
        state: jax.Array,
        rng: chex.PRNGKey
    ):
        sampled_actions = self._get_action_samples(proposal_params, state, rng)
        eval_over_proposals = jax.vmap(self._ensemble_state_action_eval, in_axes=(None, None, 0))
        q_vals = eval_over_proposals(critic_params, state, sampled_actions)
        return sampled_actions, q_vals

    def _get_top_ranked_actions(self, percentile: float, q_vals: jax.Array, sampled_actions: jax.Array):
        top_k = int(percentile * self.num_samples)
        _, top_k_inds = jax.lax.top_k(q_vals.flatten(), top_k)

        return sampled_actions[top_k_inds]

    def _get_policy_update_actions(
        self,
        critic_params: chex.ArrayTree,
        proposal_params: chex.ArrayTree,
        state: jax.Array,
        rng: chex.PRNGKey,
    ):
        sampled_actions, q_vals = self._evaluate_sampled_actions(critic_params, proposal_params, state, rng)
        actor_update_actions = self._get_top_ranked_actions(self.actor_percentile, q_vals, sampled_actions)
        proposal_update_actions = self._get_top_ranked_actions(self.proposal_percentile, q_vals, sampled_actions)

        return UpdateActions(actor_update_actions, proposal_update_actions)

    def _policy_loss(self, params: chex.ArrayTree, policy: hk.Transformed, state: jax.Array, top_actions: jax.Array):
        out: ActorOutputs = policy.apply(params=params, x=state)
        dist = distrax.Beta(out.alpha, out.beta)
        log_prob = dist.log_prob(top_actions)

        return -cast(jax.Array, log_prob)

    def _batch_policy_loss(
        self,
        params: chex.ArrayTree,
        policy: hk.Transformed,
        states: jax.Array,
        top_actions_batch: jax.Array
    ):
        losses = jax.vmap(self._policy_loss, in_axes=(None, None, 0, 0))(params, policy, states, top_actions_batch)
        return jnp.mean(losses)

    def _actor_update(
        self,
        actor_state: PolicyState,
        states: jax.Array,
        update_actions: jax.Array
    ):
        params = actor_state.params
        opt_state = actor_state.opt_state
        grads = jax.grad(self._batch_policy_loss)(params, self.actor, states, update_actions)
        updates, new_opt_state = self.actor_opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return PolicyState(
            new_params,
            new_opt_state,
        )

    def _proposal_update(
        self,
        proposal_state: PolicyState,
        states: jax.Array,
        update_actions: jax.Array
    ):
        proposal_params = proposal_state.params
        proposal_opt_state = proposal_state.opt_state
        grads = jax.grad(self._batch_policy_loss)(proposal_params, self.proposal, states, update_actions)
        updates, updated_proposal_opt_state = self.proposal_opt.update(grads, proposal_opt_state)
        updated_proposal_params = optax.apply_updates(proposal_params, updates)

        return PolicyState(
            updated_proposal_params,
            updated_proposal_opt_state,
        )

    @jax_u.method_jit
    def _policy_update(
        self,
        agent_state: GACState,
        states: jax.Array,
        rng: chex.PRNGKey,
    ):
        top_ranked_actions_over_batch = jax.vmap(self._get_policy_update_actions, in_axes=(None, None, 0, None))
        top_ranked_actions = top_ranked_actions_over_batch(
            agent_state.critic.params,
            agent_state.proposal.params,
            states,
            rng,
        )

        # Actor Update
        new_actor_state = self._actor_update(agent_state.actor, states, top_ranked_actions.actor)

        # Proposal Update
        new_proposal_state = self._proposal_update(agent_state.proposal, states, top_ranked_actions.proposal)
        # new_proposal_state = self.agent_state.proposal

        return new_actor_state, new_proposal_state

    def policy_update(self):
        if self.policy_buffer.size == 0:
            return

        batch = self.policy_buffer.sample()
        self.rng, update_rng = jax.random.split(self.rng, 2)
        actor_state, proposal_state = self._policy_update(self.agent_state, batch.state[0], update_rng)

        self.agent_state = self.agent_state._replace(
            actor=actor_state,
            proposal=proposal_state,
        )

    def update(self):
        self.critic_update()

        # print('before: ', jax.tree_util.tree_map(lambda x: float(jnp.linalg.norm(x)), self.agent_state.actor.params))
        self.policy_update()
        # print('after: ', jax.tree_util.tree_map(lambda x: float(jnp.linalg.norm(x)), self.agent_state.actor.params))
