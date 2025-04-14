from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

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
from agent.components.buffer import EnsembleReplayBuffer
from agent.components.networks.activations import (
    ActivationConfig,
    IdentityConfig,
    get_output_activation,
)
from agent.components.q_critic import SARSAConfig, SARSACritic
from interaction.transition_creator import Transition


class SquashedGaussian:
    def __init__(self, mean: jax.Array, std: jax.Array):
        dist = distrax.Transformed(
            distribution=distrax.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=distrax.Block(
                distrax.Tanh(),
                ndims=1,
            )
        )
        dist = distrax.Transformed(
            distribution=dist,
            bijector=distrax.Block(
                distrax.ScalarAffine(shift=1, scale=1.0),
                ndims=1,
            )
        )
        self.dist = distrax.Transformed(
            distribution=dist,
            bijector=distrax.Block(
                distrax.ScalarAffine(shift=0, scale=0.5),
                ndims=1,
            )
        )

    def sample(self, seed: chex.PRNGKey):
        return self.dist.sample(seed=seed)

    def log_prob(self, action: jax.Array):
        return self.dist.log_prob(action)

    def prob(self, action: jax.Array):
        return self.dist.prob(action)

class UpdateActions(NamedTuple):
    actor: jax.Array
    proposal: jax.Array


class CriticState(Protocol):
    @property
    def params(self) -> chex.ArrayTree: ...


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

    actor_lr: float
    proposal_lr: float

    critic: dict[str, Any]

class ActorOutputs(NamedTuple):
    mu: jax.Array
    sigma: jax.Array

def actor_builder(cfg: nets.TorsoConfig, act_cfg: ActivationConfig, act_dim: int):
    def _inner(x: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x)
        output_act = get_output_activation(act_cfg)
        mu_head_out = output_act(hk.Linear(act_dim)(phi))
        sigma_head_out = output_act(hk.Linear(act_dim)(phi))

        return ActorOutputs(
            mu=mu_head_out,
            sigma=jnp.exp(jnp.clip(sigma_head_out, -20, 2))
        )

    return hk.without_apply_rng(hk.transform(_inner))


class GACCritic(Protocol):
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array) -> CriticState: ...
    def forward(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array) -> jax.Array: ...
    def update(self, state: CriticState, get_actions: Callable[[chex.PRNGKey, jax.Array], jax.Array]) -> CriticState:
        ...
    def update_buffer(self, transition: Transition) -> None: ...


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

        self._collector = collector

        self._critic: GACCritic = SARSACritic(SARSAConfig(**cfg.critic), seed, state_dim, action_dim, collector)

        actor_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=256, activation='relu'),
                nets.LinearConfig(size=256, activation='relu'),
            ]
        )
        actor_output_act_cfg = IdentityConfig()
        self.actor = actor_builder(actor_torso_cfg, actor_output_act_cfg, action_dim)
        self.proposal = self.actor

        # Optimizers
        self.actor_opt = optax.adam(cfg.actor_lr)
        self.proposal_opt = optax.adam(cfg.proposal_lr)

        # Replay Buffers
        self.policy_buffer = EnsembleReplayBuffer(
            n_ensemble=1,
            ensemble_prob=1.0,
            batch_size=cfg.batch_size,
        )

        # Agent State
        dummy_x = jnp.zeros(self.state_dim)
        dummy_a = jnp.zeros(self.action_dim)

        self.rng, c_rng = jax.random.split(self.rng)
        critic_state = self._critic.init_state(c_rng, dummy_x, dummy_a)

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
    def update_buffer(self, transition: Transition):
        self._critic.update_buffer(transition)
        self.policy_buffer.add(transition)

    def _get_dist(self, params: chex.ArrayTree, state: jax.Array):
        out: ActorOutputs = self.actor.apply(params=params, x=state)
        dist = SquashedGaussian(out.mu, out.sigma)
        return dist

    @jax_u.method_jit
    def _get_actions(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array):
        dist = self._get_dist(params, state)
        return dist.sample(seed=rng)

    def get_actions(self, state: jax.Array | np.ndarray):
        state = jnp.asarray(state)
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        action = self._get_actions(self.agent_state.actor.params, sample_rng, state)
        
        entropy = self._calculate_policy_entropy(self.agent_state.actor.params, state[None])
        return action

    def _calculate_policy_entropy(self, actor_params: chex.ArrayTree, states: jax.Array):
        def entropy_fn(state: jax.Array):
            out: ActorOutputs = self.actor.apply(params=actor_params, x=state)
            dist = SquashedGaussian(out.mu, out.sigma)
            # for Gaussian, entropy is 0.5 * log(2πe * det(sigma))
            # det(sigma) = product of diagonal elements
            log_det = jnp.sum(jnp.log(out.sigma))
            entropy = 0.5 * (self.action_dim * (1.0 + jnp.log(2 * jnp.pi)) + log_det)
            return entropy
        
        entropies = jax.vmap(entropy_fn)(states)
        mean_entropy = jnp.mean(entropies)
        self._collector.collect('policy_entropy', float(mean_entropy))
        return mean_entropy

    def get_action_values(self, state: jax.Array | np.ndarray, actions: jax.Array | np.ndarray):
        return self._critic.forward(
            self.agent_state.critic.params,
            x=jnp.asarray(state),
            a=jnp.asarray(actions),
        )

    def _get_prob(self, dist: SquashedGaussian, action: jax.Array):
        log_prob = dist.log_prob(action)
        return jnp.exp(log_prob)

    def _get_probs(self, dist: SquashedGaussian, actions: jax.Array):
        probs = jax.vmap(self._get_prob, in_axes=(None, 0))(dist, actions)
        return probs

    @jax_u.method_jit
    def get_probs(self, params: chex.ArrayTree, state: jax.Array | np.ndarray, actions: jax.Array):
        state = jnp.asarray(state)
        dist = self._get_dist(params, state)
        probs = self._get_probs(dist, actions)

        return probs

    def update(self):
        self.critic_update()
        self.policy_update()


    # -------------------
    # -- Critic Update --
    # -------------------
    @jax_u.method_jit
    def _get_actions_over_state(self, actor_params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array):
        chex.assert_rank(x, 3)
        return jax_u.vmap_only(self._get_actions, ['state'])(
            actor_params,
            rng,
            x,
        )

    def critic_update(self):
        new_critic_state = self._critic.update(
            state=self.agent_state.critic,
            get_actions=lambda rng, x: self._get_actions_over_state(
                self.agent_state.actor.params,
                rng,
                x,
            )
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)


    # --------------------
    # -- Policy Updates --
    # --------------------
    def policy_update(self):
        if self.policy_buffer.size == 0:
            return

        batch = self.policy_buffer.sample()
        self.rng, update_rng = jax.random.split(self.rng, 2)
        actor_state, proposal_state, actor_loss = self._policy_update(self.agent_state, batch.state[0], update_rng)
        
        # Log actor loss outside the JIT-compiled function
        self._collector.collect("actor_loss", float(actor_loss))

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
        new_actor_state, actor_loss = self._compute_policy_update(
            agent_state.actor,
            self.actor,
            self.actor_opt,
            states,
            top_ranked_actions.actor,
        )

        # Proposal Update
        new_proposal_state, _ = self._compute_policy_update(
            agent_state.proposal,
            self.proposal,
            self.proposal_opt,
            states,
            top_ranked_actions.proposal,
        )
        return new_actor_state, new_proposal_state, actor_loss

    def _compute_policy_update(
        self,
        policy_state: PolicyState,
        policy: hk.Transformed,
        policy_opt: optax.GradientTransformation,
        states: jax.Array,
        update_actions: jax.Array
    ):
        loss_fn = lambda params: self._batch_policy_loss(params, policy, states, update_actions)
        loss_value = loss_fn(policy_state.params)
        grads = jax.grad(loss_fn)(policy_state.params)
        updates, new_opt_state = policy_opt.update(grads, policy_state.opt_state)
        new_params = optax.apply_updates(policy_state.params, updates)
        
        return PolicyState(
            new_params,
            new_opt_state,
        ), loss_value

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

        q_over_ens = jax_u.vmap_only(self._critic.forward, ['params'])
        q_over_proposal = jax_u.vmap_only(q_over_ens, ['a'])
        ens_q_vals = q_over_proposal(critic_params, state, proposal_actions)
        chex.assert_shape(ens_q_vals, (self._cfg.num_samples, self._cfg.critic['ensemble'], 1))

        q_vals = ens_q_vals.mean(axis=1)[:, 0]
        chex.assert_shape(q_vals, (self._cfg.num_samples, ))

        actor_k = int(self._cfg.actor_percentile * self._cfg.num_samples)
        actor_update_actions = top_k_by_other(proposal_actions, q_vals, actor_k)

        proposal_k = int(self._cfg.proposal_percentile * self._cfg.num_samples)
        proposal_update_actions = top_k_by_other(proposal_actions, q_vals, proposal_k)

        return UpdateActions(actor_update_actions, proposal_update_actions)

    def _policy_loss(self, params: chex.ArrayTree, policy: hk.Transformed, state: jax.Array, top_actions: jax.Array):
        out: ActorOutputs = policy.apply(params=params, x=state)
        dist = SquashedGaussian(out.mu, out.sigma)
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
