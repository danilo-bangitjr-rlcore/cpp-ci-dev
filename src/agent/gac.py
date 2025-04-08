from dataclasses import dataclass
from typing import Any, NamedTuple

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
    name: str
    num_samples: int
    actor_percentile: float
    proposal_percentile: float
    uniform_weight: float
    batch_size: int
    ensemble: int


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
        self._cfg = cfg
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
        critic_lr = 0.1
        self.critic_opt = optax.chain(
            optax.adam(learning_rate=critic_lr),
            optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=50,
                max_learning_rate=critic_lr,
                decrease_factor=0.9,
                increase_factor=np.inf,
                slope_rtol=0.1
            ),
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
        critic_state = jax_u.vmap_only(self.init_critic_state, ['rng'])(rngs, dummy_x, dummy_a)

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

    # ----------------------
    # -- Public Interface --
    # ----------------------

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

    def _get_dist(self, params: chex.ArrayTree, state: jax.Array) -> distrax.Distribution:
        out: ActorOutputs = self.actor.apply(params=params, x=state)
        dist = distrax.Beta(out.alpha, out.beta)
        return dist

    @jax_u.method_jit
    def _get_actions(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array):
        dist = self._get_dist(params, state)
        return dist.sample(seed=rng)

    def get_actions(self, state: jax.Array | np.ndarray):
        state = jnp.asarray(state)
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        return self._get_actions(self.agent_state.actor.params, sample_rng, state)

    def _get_prob(self, dist: distrax.Distribution, action: jax.Array):
        log_prob = dist.log_prob(action)
        return jnp.exp(log_prob)

    def _get_probs(self, dist: distrax.Distribution, actions: jax.Array):
        probs = jax.vmap(self._get_prob, in_axes=(None, 0))(dist, actions)
        return probs

    @jax_u.method_jit
    def get_probs(self, params: chex.ArrayTree, state: jax.Array | np.ndarray, actions: jax.Array):
        state = jnp.asarray(state)
        dist = self._get_dist(params, state)
        probs = self._get_probs(dist, actions)

        return probs

    def _get_action_value(self, params: chex.ArrayTree, state: jax.Array, action: jax.Array):
        return self.critic.apply(params=params, x=state, a=action).q

    def _get_ensemble_action_value(self, params: chex.ArrayTree, state: jax.Array, action: jax.Array):
        ensemble_q_vals = jax.vmap(self._get_action_value, in_axes=(0, None, None))(params, state, action)
        q_val = jnp.mean(ensemble_q_vals) # Mean Reduction

        return q_val

    @jax_u.method_jit
    def get_action_values(self, params: chex.ArrayTree, state: jax.Array | np.ndarray, actions: jax.Array):
        state = jnp.asarray(state)
        q_values = jax.vmap(self._get_ensemble_action_value, in_axes=(None, None, 0))(params, state, actions)

        return q_values

    def update(self):
        self.critic_update()
        self.policy_update()


    # -------------------
    # -- Critic Update --
    # -------------------
    def _get_ensemble_values(self, params: chex.ArrayTree, state: jax.Array, action: jax.Array) -> jax.Array:
        apply = jax_u.vmap_only(self.critic.apply, ['params'])
        return apply(params, state, action).q

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
        target_value = self._get_ensemble_values(ens_target_params, next_state, next_action).mean(axis=0)
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
        loss_over_batch = jax_u.vmap_only(self.critic_loss, ['transition', 'rng'])
        losses = loss_over_batch(
            critic_params,
            ens_target_params,
            actor_params,
            transitions,
            rngs,
        )

        return jnp.mean(losses)

    def _ens_critic_loss(
        self,
        params: chex.ArrayTree,
        ens_target_params: chex.ArrayTree,
        actor_params: chex.ArrayTree,
        transitions: VectorizedTransition,
        rng: chex.PRNGKey,
    ):
        rngs = jax.random.split(rng, self.ensemble)
        loss_over_members = jax_u.vmap_only(self._batch_critic_loss, ['critic_params', 'transitions', 'rng'])
        losses = loss_over_members(params, ens_target_params, actor_params, transitions, rngs)
        return jnp.sum(losses), losses

    @jax_u.method_jit
    def _ensemble_critic_update(
        self,
        critic_state: CriticState,
        actor_params: chex.ArrayTree,
        transitions: VectorizedTransition,
        rng: chex.PRNGKey,
    ):
        """
        Updates each member of the ensemble.
        """
        grads, member_losses = jax.grad(self._ens_critic_loss, has_aux=True)(
            critic_state.params,
            critic_state.target_params,
            actor_params,
            transitions,
            rng,
        )

        ens_updates = []
        ens_opts = []
        for i in range(self._cfg.ensemble):
            updates, new_opt_state = self.critic_opt.update(
                get_member(grads, i),
                get_member(critic_state.opt_state, i),
                get_member(critic_state.params, i),
                value=member_losses[i],
                grad=get_member(grads, i),
                value_fn=self._batch_critic_loss,
                ens_target_params=critic_state.target_params,
                actor_params=actor_params,
                transitions=get_member(transitions, i),
                rng=rng,
            )

            ens_updates.append(updates)
            ens_opts.append(new_opt_state)

        updates = jax.tree_util.tree_map(lambda *upd: jnp.stack(upd, axis=0), *ens_updates)
        new_opt_state = jax.tree_util.tree_map(lambda *opt: jnp.stack(opt, axis=0), *ens_opts)
        new_params = optax.apply_updates(critic_state.params, updates)
        # Target Net Polyak Update
        polyak = 0.995
        target_params = critic_state.target_params
        new_target_params = optax.incremental_update(new_params, target_params, 1 - polyak)

        return CriticState(
            new_params,
            new_target_params,
            new_opt_state,
        ), member_losses

    def critic_update(self):
        if self.critic_buffer.size == 0:
            return
        transitions = self.critic_buffer.sample()

        self.rng, update_rng = jax.random.split(self.rng, 2)
        new_critic_state, losses = self._ensemble_critic_update(
            self.agent_state.critic,
            self.agent_state.actor.params,
            transitions,
            update_rng,
        )
        self.agent_state = self.agent_state._replace(critic=new_critic_state)

        loss = jnp.mean(losses)
        self._collector.collect('critic_loss', float(loss))

    # --------------------
    # -- Policy Updates --
    # --------------------
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

    @jax_u.method_jit
    def _policy_update(
        self,
        agent_state: GACState,
        states: jax.Array,
        rng: chex.PRNGKey,
    ):
        chex.assert_shape(states, (self._cfg.batch_size, self.state_dim))

        top_ranked_actions_over_batch = jax_u.vmap_only(self._get_policy_update_actions, ['state'])
        top_ranked_actions = top_ranked_actions_over_batch(
            agent_state.critic.params,
            agent_state.proposal.params,
            states,
            rng,
        )

        # Actor Update
        new_actor_state = self._compute_policy_update(
            agent_state.actor,
            self.actor,
            self.actor_opt,
            states,
            top_ranked_actions.actor,
        )

        # Proposal Update
        new_proposal_state = self._compute_policy_update(
            agent_state.proposal,
            self.proposal,
            self.proposal_opt,
            states,
            top_ranked_actions.proposal,
        )
        return new_actor_state, new_proposal_state

    def _compute_policy_update(
        self,
        policy_state: PolicyState,
        policy: hk.Transformed,
        policy_opt: optax.GradientTransformation,
        states: jax.Array,
        update_actions: jax.Array
    ):
        grads = jax.grad(self._batch_policy_loss)(policy_state.params, policy, states, update_actions)
        updates, new_opt_state = policy_opt.update(grads, policy_state.opt_state)
        new_params = optax.apply_updates(policy_state.params, updates)

        return PolicyState(
            new_params,
            new_opt_state,
        )

    def _get_proposal_samples(self, proposal_params: chex.ArrayTree, state: jax.Array, rng: chex.PRNGKey):
        uniform_samples = int(self.num_samples * self.uniform_weight)

        rng, u_rng = jax.random.split(rng, 2)
        uniform_actions = jax.random.uniform(u_rng, (uniform_samples, self.action_dim))

        proposal_samples = self.num_samples - uniform_samples
        if proposal_samples == 0:
            return uniform_actions

        rngs = jax.random.split(self.rng, proposal_samples)
        proposal_actions = jax_u.vmap_only(self._get_actions, ['rng'])(proposal_params, rngs, state)

        sampled_actions = jnp.concat([uniform_actions, proposal_actions], axis=0)
        return sampled_actions

    def _get_policy_update_actions(
        self,
        critic_params: chex.ArrayTree,
        proposal_params: chex.ArrayTree,
        state: jax.Array,
        rng: chex.PRNGKey,
    ):
        chex.assert_shape(state, (self.state_dim, ))

        proposal_actions = self._get_proposal_samples(proposal_params, state, rng)
        chex.assert_shape(proposal_actions, (self._cfg.num_samples, self.action_dim))

        ens_q_vals = jax_u.vmap_only(self._get_ensemble_values, ['action'])(
            critic_params,
            state,
            proposal_actions,
        )
        chex.assert_shape(ens_q_vals, (self._cfg.num_samples, self._cfg.ensemble, 1))

        q_vals = ens_q_vals.mean(axis=1)[:, 0]
        chex.assert_shape(q_vals, (self._cfg.num_samples, ))

        actor_k = int(self._cfg.actor_percentile * self._cfg.num_samples)
        actor_update_actions = top_k_by_other(proposal_actions, q_vals, actor_k)

        proposal_k = int(self._cfg.proposal_percentile * self._cfg.num_samples)
        proposal_update_actions = top_k_by_other(proposal_actions, q_vals, proposal_k)

        return UpdateActions(actor_update_actions, proposal_update_actions)

    def _policy_loss(self, params: chex.ArrayTree, policy: hk.Transformed, state: jax.Array, top_actions: jax.Array):
        out: ActorOutputs = policy.apply(params=params, x=state)
        dist = distrax.Beta(out.alpha, out.beta)
        log_prob = dist.log_prob(top_actions) # log prob for each action dimension
        loss = jnp.sum(log_prob)

        return -loss

    def _batch_policy_loss(
        self,
        params: chex.ArrayTree,
        policy: hk.Transformed,
        states: jax.Array,
        top_actions_batch: jax.Array
    ):
        losses = jax.vmap(self._policy_loss, in_axes=(None, None, 0, 0))(params, policy, states, top_actions_batch)
        return jnp.mean(losses)


def top_k_by_other(arr: jax.Array, other: jax.Array, k: int) -> jax.Array:
    chex.assert_equal_shape_prefix((arr, other), prefix_len=1)
    chex.assert_axis_dimension_gteq(arr, 0, k)
    chex.assert_axis_dimension_gteq(other, 0, k)

    _, idxs = jax.lax.top_k(other, k)
    return arr[idxs]


def get_member(a: Any, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], a)
