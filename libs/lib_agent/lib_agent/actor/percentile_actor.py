from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Protocol

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax
from ml_instrumentation.Collector import Collector

import lib_agent.network.networks as nets
from lib_agent.buffer.buffer import State
from lib_agent.network.activations import (
    ActivationConfig,
    IdentityConfig,
    get_output_activation,
)


class SquashedGaussian:
    def __init__(self, mean: jax.Array, std: jax.Array):
        dist = distrax.Transformed(
            distribution=distrax.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=distrax.Block(
                distrax.Tanh(),
                ndims=1,
            ),
        )
        dist = distrax.Transformed(
            distribution=dist,
            bijector=distrax.Block(
                distrax.ScalarAffine(shift=1, scale=1.0),
                ndims=1,
            ),
        )
        self.dist = distrax.Transformed(
            distribution=dist,
            bijector=distrax.Block(
                distrax.ScalarAffine(shift=0, scale=0.5),
                ndims=1,
            ),
        )
        self.mean = mean
        self.std = std

    def sample(self, seed: chex.PRNGKey):
        return self.dist.sample(seed=seed)

    def log_prob(self, action: jax.Array):
        return self.dist.log_prob(action)

    def prob(self, action: jax.Array):
        return self.dist.prob(action)

    def get_params(self):
        return {
            'mean': self.mean,
            'std': self.std,
        }

class UpdateActions(NamedTuple):
    actor: jax.Array
    proposal: jax.Array

class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree

class PAState(NamedTuple):
    actor: PolicyState
    proposal: PolicyState


class ActorBatch(Protocol):
    @property
    def state(self) -> State: ...
    @property
    def action(self) -> jax.Array: ...
    @property
    def reward(self) -> jax.Array: ...
    @property
    def next_state(self) -> State: ...
    @property
    def gamma(self) -> jax.Array: ...

class ValueEstimator(Protocol):
    def __call__(
        self,
        params: chex.ArrayTree,
        x: jax.Array,
        a: jax.Array,
    ) -> jax.Array: ...


@dataclass
class PAConfig:
    name: str
    num_samples: int
    actor_percentile: float
    proposal_percentile: float
    uniform_weight: float
    actor_lr: float
    proposal_lr: float
    max_action_stddev: float | None = None
    sort_noise: float = 0.0

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
            sigma=jnp.exp(jnp.clip(sigma_head_out, -20, 2)),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class PercentileActor:
    def __init__(self, cfg: PAConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._cfg = cfg

        self._collector = collector

        actor_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=256, activation='relu'),
                nets.LinearConfig(size=256, activation='relu'),
            ],
        )
        actor_output_act_cfg = IdentityConfig()
        self.actor = actor_builder(actor_torso_cfg, actor_output_act_cfg, action_dim)
        self.proposal = self.actor

        self.actor_opt = optax.adam(cfg.actor_lr)
        self.proposal_opt = optax.adam(cfg.proposal_lr)

    # -------------------------------- Init State -------------------------------- #

    @jax_u.method_jit
    def init_state(self, rng: chex.PRNGKey, x: jax.Array):
        rng_1, rng_2 = jax.random.split(rng)
        actor_params = self.actor.init(rng=rng_1, x=x)
        actor_state = PolicyState(
            params=actor_params,
            opt_state=self.actor_opt.init(actor_params),
        )

        proposal_params = self.proposal.init(rng=rng_2, x=x)
        proposal_state = PolicyState(
            params=proposal_params,
            opt_state=self.proposal_opt.init(proposal_params),
        )

        return PAState(actor_state, proposal_state)

    # -------------------------------- get actions ------------------------------- #

    def safe_get_actions(self, actor_params: chex.ArrayTree, state: State):
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        return self.safe_get_actions_rng(actor_params, sample_rng, state)

    def safe_get_actions_rng(self, actor_params: chex.ArrayTree, rng: chex.PRNGKey,  state: State):
        dist = self._get_dist(actor_params, state)
        params = dist.get_params()
        mean, std = params['mean'], params['std']

        sampled = dist.sample(seed=rng)
        clipped_to_bounds = jnp.clip(sampled, state.a_lo, state.a_hi)

        if self._cfg.max_action_stddev is None:
            return clipped_to_bounds

        return jnp.clip(
            clipped_to_bounds,
            mean-std*self._cfg.max_action_stddev,
            mean+std*self._cfg.max_action_stddev,
        )

    def get_actions(self, actor_params: chex.ArrayTree, state: State):
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        return self.get_actions_rng(actor_params, sample_rng, state)

    @jax_u.method_jit
    def get_actions_rng(self, actor_params: chex.ArrayTree, rng: chex.PRNGKey, state: State):
        dist = self._get_dist(actor_params, state)
        sampled = dist.sample(seed=rng)
        dp_mask = state.dp
        clipped = jnp.clip(sampled, state.a_lo, state.a_hi)
        return dp_mask * clipped + (1 - dp_mask) * state.last_a

    def _get_dist(self, actor_params: chex.ArrayTree, state: State):
        out: ActorOutputs = self.actor.apply(params=actor_params, x=state.features)
        return SquashedGaussian(out.mu, out.sigma)

    # ----------------------------- get probabilities ---------------------------- #

    @jax_u.method_jit
    def get_probs(self, params: chex.ArrayTree, state: State, actions: jax.Array):
        dist = self._get_dist(params, state)
        return self._get_probs(dist, actions)

    def _get_prob(self, dist: SquashedGaussian, action: jax.Array):
        log_prob = dist.log_prob(action)
        return jnp.exp(log_prob)

    def _get_probs(self, dist: SquashedGaussian, actions: jax.Array):
        return jax.vmap(self._get_prob, in_axes=(None, 0))(dist, actions)

    # ---------------------------------- updates --------------------------------- #

    def update(
        self,
        pa_state: Any,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        transitions: ActorBatch,
    ):

        self.rng, update_rng = jax.random.split(self.rng, 2)

        states = jax.tree.map(lambda arr: arr[0], transitions.state) # remove ensemble dimension
        actor_state, proposal_state, actor_loss = self._policy_update(
            pa_state,
            value_estimator,
            value_estimator_params,
            states,
            update_rng,
        )

        # Log actor loss outside the JIT-compiled function
        self._collector.collect("actor_loss", float(actor_loss))

        return PAState(actor_state, proposal_state)

    @partial(jax.jit, static_argnums=(0, 2))
    def _policy_update(
        self,
        pa_state: PAState,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        states: State,
        rng: chex.PRNGKey,
    ):

        sample_state = states.features[0]
        actor_out: ActorOutputs = self.actor.apply(params=pa_state.actor.params, x=sample_state)

        for i in range(actor_out.mu.shape[0]):
            jax.debug.callback(lambda x, i=i: self._collector.collect(f'actor_mu_{i}', float(x)), actor_out.mu[i])
            jax.debug.callback(lambda x, i=i: self._collector.collect(f'actor_sigma_{i}', float(x)), actor_out.sigma[i])

        top_ranked_actions_over_batch = jax_u.vmap_only(self._get_policy_update_actions, ['state'])
        top_ranked_actions = top_ranked_actions_over_batch(
            value_estimator,
            value_estimator_params,
            pa_state.proposal.params,
            states,
            rng,
        )

        # Actor Update
        new_actor_state, actor_loss = self._compute_policy_update(
            pa_state.actor,
            self.actor,
            self.actor_opt,
            states,
            top_ranked_actions.actor,
        )

        # Proposal Update
        new_proposal_state, _ = self._compute_policy_update(
            pa_state.proposal,
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
        states: State,
        update_actions: jax.Array,
    ):
        is_actor = policy is self.actor
        prefix = "actor" if is_actor else "proposal"

        def loss_fn(params: chex.ArrayTree):
            loss = self._batch_policy_loss(params, policy, states, update_actions)
            jax.debug.callback(lambda x: self._collector.collect(f'{prefix}_loss', float(x)), loss)
            return loss

        loss_value = loss_fn(policy_state.params)
        grads = jax.grad(loss_fn)(policy_state.params)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        if grad_leaves:
            grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grad_leaves))
            jax.debug.callback(lambda x: self._collector.collect(f'{prefix}_grad_norm', float(x)), grad_norm)

        updates, new_opt_state = policy_opt.update(grads, policy_state.opt_state)
        new_params = optax.apply_updates(policy_state.params, updates)

        return PolicyState(
            new_params,
            new_opt_state,
        ), loss_value

    def _get_proposal_samples(self, proposal_params: chex.ArrayTree, state: State, rng: chex.PRNGKey):
        uniform_samples = int(self._cfg.num_samples * self._cfg.uniform_weight)

        rng, u_rng = jax.random.split(rng, 2)
        uniform_actions = jax.random.uniform(u_rng, (uniform_samples, self.action_dim))
        uniform_actions = jnp.clip(uniform_actions, state.a_lo, state.a_hi)

        proposal_samples = self._cfg.num_samples - uniform_samples
        if proposal_samples == 0:
            return uniform_actions

        rngs = jax.random.split(self.rng, proposal_samples)
        proposal_actions = jax_u.vmap_only(self.get_actions_rng, ['rng'])(proposal_params, rngs, state)

        return jnp.concat([uniform_actions, proposal_actions], axis=0)

    def _get_policy_update_actions(
        self,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        proposal_params: chex.ArrayTree,
        state: State,
        rng: chex.PRNGKey,
    ):
        chex.assert_shape(state.features, (self.state_dim, ))

        sample_rng, rng = jax.random.split(rng, 2)
        proposal_actions = self._get_proposal_samples(proposal_params, state, sample_rng)
        # clip proposal action to prevent log prob from being nan
        proposal_actions = jnp.clip(proposal_actions, 1e-5, 1 - 1e-5)
        chex.assert_shape(proposal_actions, (self._cfg.num_samples, self.action_dim))

        q_over_proposal = jax_u.vmap_only(value_estimator, ['a'])
        q_vals = q_over_proposal(value_estimator_params, state.features, proposal_actions)
        q_vals = q_vals + jax.random.normal(
            rng, shape=q_vals.shape, dtype=q_vals.dtype,
        ) * self._cfg.sort_noise
        chex.assert_shape(q_vals, (self._cfg.num_samples, ))

        actor_k = int(self._cfg.actor_percentile * self._cfg.num_samples)
        actor_update_actions = top_k_by_other(proposal_actions, q_vals, actor_k)

        proposal_k = int(self._cfg.proposal_percentile * self._cfg.num_samples)
        proposal_update_actions = top_k_by_other(proposal_actions, q_vals, proposal_k)

        return UpdateActions(actor_update_actions, proposal_update_actions)

    def _policy_loss(self, params: chex.ArrayTree, policy: hk.Transformed, state: State, top_actions: jax.Array):
        out: ActorOutputs = policy.apply(params=params, x=state.features)
        dist = SquashedGaussian(out.mu, out.sigma)
        log_prob = dist.log_prob(top_actions) # log prob for each action dimension
        loss = jnp.sum(log_prob)

        return -loss

    def _batch_policy_loss(
        self,
        params: chex.ArrayTree,
        policy: hk.Transformed,
        states: State,
        top_actions_batch: jax.Array,
    ):
        losses = jax.vmap(self._policy_loss, in_axes=(None, None, 0, 0))(params, policy, states, top_actions_batch)
        return jnp.mean(losses)


def top_k_by_other(arr: jax.Array, other: jax.Array, k: int) -> jax.Array:
    chex.assert_equal_shape_prefix((arr, other), prefix_len=1)
    chex.assert_axis_dimension_gteq(arr, 0, k)
    chex.assert_axis_dimension_gteq(other, 0, k)

    _, idxs = jax.lax.top_k(other, k)
    return arr[idxs]
