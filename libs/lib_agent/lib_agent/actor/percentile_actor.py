from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import lib_utils.parameter_groups as param_groups
import optax

import lib_agent.network.networks as nets
from lib_agent.actor.actor_protocol import ActorConfig, ActorUpdateMetrics, PolicyOutputs, PolicyState, ValueEstimator
from lib_agent.buffer.datatypes import State, Transition
from lib_agent.network.activations import (
    ActivationConfig,
    IdentityConfig,
    get_output_activation,
)


class UpdateActions(NamedTuple):
    actor: jax.Array
    proposal: jax.Array


class PAState(NamedTuple):
    actor: PolicyState
    proposal: PolicyState


@dataclass
class PAConfig(ActorConfig):
    num_samples: int = 128
    actor_percentile: float = 0.05
    proposal_percentile: float = 0.2
    uniform_weight: float = 0.8
    actor_lr: float = 0.0001
    proposal_lr: float = 0.0001
    sort_noise: float = 0.0
    mu_multiplier: float = 1.0
    sigma_multiplier: float = 1.0
    max_action_stddev: float = jnp.inf


def actor_builder(cfg: nets.TorsoConfig, act_cfg: ActivationConfig, act_dim: int):
    def _inner(x: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x)
        output_act = get_output_activation(act_cfg)
        mu_head_out = output_act(hk.Linear(act_dim, name='mu_head')(phi))
        sigma_head_out = output_act(hk.Linear(act_dim, name='sigma_head')(phi))

        return PolicyOutputs(
            mu=mu_head_out,
            sigma=jax.nn.sigmoid(sigma_head_out) * 0.1 + 1e-5,
        )

    return hk.without_apply_rng(hk.transform(_inner))


class PercentileActor:
    def __init__(self, cfg: PAConfig, seed: int, state_dim: int, action_dim: int):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._cfg = cfg

        actor_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=128, activation='crelu'),
                nets.LinearConfig(size=128, activation='crelu'),
            ],
        )
        actor_output_act_cfg = IdentityConfig()
        self.actor = actor_builder(actor_torso_cfg, actor_output_act_cfg, action_dim)
        self.proposal = self.actor

        self.actor_opt = optax.adamw(cfg.actor_lr, weight_decay=0.001)
        self.proposal_opt = optax.adamw(cfg.proposal_lr, weight_decay=0.001)

        self._actor_param_manager = self._create_mu_sigma_groups(
            mu_lr=cfg.actor_lr * cfg.mu_multiplier,
            sigma_lr=cfg.actor_lr * cfg.sigma_multiplier,
            shared_lr=cfg.actor_lr,
        )

        self._proposal_param_manager = self._create_mu_sigma_groups(
            mu_lr=cfg.proposal_lr * cfg.mu_multiplier,
            sigma_lr=cfg.proposal_lr * cfg.sigma_multiplier,
            shared_lr=cfg.proposal_lr,
        )

    def _create_mu_sigma_groups(
        self,
        mu_lr: float,
        sigma_lr: float,
        shared_lr: float,
        weight_decay: float = 0.001,
    ) -> param_groups.ParameterGroupManager:
        manager = param_groups.ParameterGroupManager()
        manager.add_group(
            'mu',
            ['mu_head'],
            optax.adamw(mu_lr, weight_decay=weight_decay),
        )
        manager.add_group(
            'sigma',
            ['sigma_head'],
            optax.adamw(sigma_lr, weight_decay=weight_decay),
        )
        manager.add_group(
            'shared',
            ['linear'],
            optax.adamw(shared_lr, weight_decay=weight_decay),
        )
        return manager

    # -------------------------------- Init State -------------------------------- #

    @jax_u.method_jit
    def init_state(self, rng: chex.PRNGKey, x: jax.Array):
        rng_1, rng_2 = jax.random.split(rng)
        actor_params = self.actor.init(rng=rng_1, x=x)
        actor_group_opt_states = self._actor_param_manager.init_optimizer_states(actor_params)
        actor_state = PolicyState(
            params=actor_params,
            opt_state=None,
            group_opt_states=actor_group_opt_states,
        )
        proposal_params = self.proposal.init(rng=rng_2, x=x)
        proposal_group_opt_states = self._proposal_param_manager.init_optimizer_states(proposal_params)
        proposal_state = PolicyState(
            params=proposal_params,
            opt_state=None,
            group_opt_states=proposal_group_opt_states,
        )
        return PAState(actor_state, proposal_state)

    @jax_u.method_jit
    def _forward(self, actor_params: chex.ArrayTree, state: State) -> PolicyOutputs:
        levels = state.features.ndim - 1
        return jax_u.vmap_only(self.actor.apply, ['x'], levels)(
            actor_params,
            state.features.array,
        )


    def initialize_to_nominal_action(
        self,
        rng: chex.PRNGKey,
        policy_state: PolicyState,
        nominal_actions: jax.Array,
        state_dim: int,
    ):
        def regress_to_nominal(
            params: chex.ArrayTree,
            rng: chex.PRNGKey,
        ):
            BATCH = 32

            states = jax.random.uniform(rng, shape=(BATCH, state_dim))
            dist_params = jax_u.vmap_only(self.actor.apply, ['x'])(
                params,
                states,
            )

            mu_loss = jnp.square(dist_params.mu - nominal_actions).mean()
            sigma_loss = jnp.square(dist_params.sigma - 0.1).mean()

            return mu_loss + sigma_loss

        @jax_u.jit
        def update_params(
            params: chex.ArrayTree,
            opt_state: chex.ArrayTree,
            rng: chex.PRNGKey,
        ):
            loss, grads = jax.value_and_grad(regress_to_nominal)(params, rng)
            updates, new_opt_state = self.actor_opt.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_opt_state

        params = policy_state.params
        opt_state = self.actor_opt.init(params)
        for _ in range(100):
            rng, update_rng = jax.random.split(rng)
            _, params, opt_state = update_params(params, opt_state, update_rng)

        return policy_state._replace(
            params=params,
        )

    # -------------------------------- get actions ------------------------------- #

    def get_actions(self, actor_params: chex.ArrayTree, state: State, n: int = 1, std_devs: float = jnp.inf):
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        return self.get_actions_rng(actor_params, sample_rng, state, n=n, std_devs=std_devs)


    @partial(jax_u.jit, static_argnums=(0, 4))
    def get_actions_rng(
        self,
        actor_params: chex.ArrayTree,
        rng: chex.PRNGKey,
        states: State,
        n: int = 1,
        std_devs: float = jnp.inf,
    ):
        chex.assert_equal_rank(states)

        # vmap over all dimensions except the last
        vmap_shape = states.features.shape[:-1]

        rngs = jax.random.split(rng, vmap_shape)
        pi = partial(self._get_actions_for_state, actor_params, n=n, std_devs=std_devs)
        actions, metrics = jax_u.multi_vmap(pi, levels=len(vmap_shape))(
            rngs,
            states,
        )

        chex.assert_shape(actions, (*vmap_shape, n, self.action_dim))
        return actions, metrics

    def _get_actions_for_state(
        self,
        actor_params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: State,
        n: int,
        std_devs: float = jnp.inf,
    ):
        chex.assert_shape(state.features, (self.state_dim, ))
        dist_params = self._forward(actor_params, state)
        dist = distrax.MultivariateNormalDiag(dist_params.mu, dist_params.sigma)
        actions = dist.sample(seed=rng, sample_shape=n)

        actions = jnp.clip(actions, state.a_lo, state.a_hi)
        actions = state.dp * actions + (1 - state.dp) * jnp.expand_dims(state.last_a, axis=0)

        actions = jnp.clip(
            actions,
            dist_params.mu - dist_params.sigma * std_devs,
            dist_params.mu + dist_params.sigma * std_devs,
        )

        chex.assert_shape(actions, (n, self.action_dim))
        return actions, {
            'actor_var': dist_params.sigma,
        }


    def get_dist(self, actor_params: chex.ArrayTree, state: State):
        dist_params = self._forward(actor_params, state)
        return distrax.MultivariateNormalDiag(dist_params.mu, dist_params.sigma)


    # ----------------------------- get probabilities ---------------------------- #

    @jax_u.method_jit
    def get_probs(self, params: chex.ArrayTree, state: State, actions: jax.Array):
        dist = self.get_dist(params, state)
        return jnp.asarray(jax_u.vmap(dist.prob)(actions))

    @jax_u.method_jit
    def get_log_probs(self, params: chex.ArrayTree, state: State, actions: jax.Array):
        dist = self.get_dist(params, state)
        return jnp.asarray(jax_u.vmap(dist.log_prob)(actions))

    # ---------------------------------- updates --------------------------------- #

    def update(
        self,
        dist_state: PAState,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        transitions: Transition,
    ):

        self.rng, update_rng = jax.random.split(self.rng, 2)

        states = jax.tree.map(lambda arr: arr[0], transitions.state) # remove ensemble dimension
        chex.assert_equal_rank(states)

        actor_state, proposal_state, metrics = self._policy_update(
            dist_state,
            value_estimator,
            value_estimator_params,
            states,
            update_rng,
        )

        return PAState(actor_state, proposal_state), metrics

    @partial(jax_u.jit, static_argnums=(0, 2))
    def _policy_update(
        self,
        pa_state: PAState,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        states: State,
        rng: chex.PRNGKey,
    ):
        top_ranked_actions_over_batch = jax_u.vmap_only(self._get_policy_update_actions, ['state'])
        top_ranked_actions = top_ranked_actions_over_batch(
            value_estimator,
            value_estimator_params,
            pa_state.proposal.params,
            states,
            rng,
        )

        # Actor Update
        new_actor_state, metrics = self._compute_policy_update(
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
        return new_actor_state, new_proposal_state, metrics

    def _compute_policy_update(
        self,
        policy_state: PolicyState,
        policy: hk.Transformed,
        policy_opt: optax.GradientTransformation,
        states: State,
        update_actions: jax.Array,
    ):
        loss, grads = jax.value_and_grad(self._batch_policy_loss)(policy_state.params, policy, states, update_actions)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grad_leaves))

        if policy_state.group_opt_states is not None:
            param_manager = self._actor_param_manager if policy is self.actor else self._proposal_param_manager
            new_params, new_opt_states = param_manager.update_parameters(
                policy_state.params, grads, policy_state.group_opt_states,
            )

            return PolicyState(
                new_params,
                None,
                new_opt_states,
            ), ActorUpdateMetrics(
                actor_loss=loss,
                actor_grad_norm=grad_norm,
            )
        if policy_state.opt_state is not None:
            updates, new_opt_state = policy_opt.update(grads, policy_state.opt_state, params=policy_state.params)
            new_params = optax.apply_updates(policy_state.params, updates)

            return PolicyState(
                new_params,
                new_opt_state,
            ), ActorUpdateMetrics(
                actor_loss=loss,
                actor_grad_norm=grad_norm,
            )
        raise ValueError("Invalid state: neither parameter groups nor standard optimizer state available")

    def _get_proposal_samples(self, proposal_params: chex.ArrayTree, state: State, rng: chex.PRNGKey):
        num_uniform_samples = int(self._cfg.num_samples * self._cfg.uniform_weight)

        rng, u_rng, p_rng = jax.random.split(rng, 3)
        uniform_samples = jax.random.uniform(u_rng, (num_uniform_samples, self.action_dim))
        uniform_actions = uniform_samples * (state.a_hi - state.a_lo) + state.a_lo

        proposal_samples = self._cfg.num_samples - num_uniform_samples
        if proposal_samples == 0:
            return uniform_actions

        proposal_actions, _ = self.get_actions_rng(proposal_params, p_rng, state, n=proposal_samples)

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

        rng, sample_rng, q_rng = jax.random.split(rng, 3)

        proposal_actions = self._get_proposal_samples(proposal_params, state, sample_rng)
        # clip proposal action to prevent log prob from being nan
        proposal_actions = jnp.clip(proposal_actions, 1e-5, 1 - 1e-5)
        chex.assert_shape(proposal_actions, (self._cfg.num_samples, self.action_dim))

        q_vals = value_estimator(value_estimator_params, q_rng, state.features.array, proposal_actions)
        chex.assert_shape(q_vals, (self._cfg.num_samples, ))
        q_vals = q_vals + self._cfg.sort_noise * jax.random.normal(
            rng, shape=q_vals.shape, dtype=q_vals.dtype,
        )
        chex.assert_shape(q_vals, (self._cfg.num_samples, ))

        # Select top-k actions based on the value estimator's output
        actor_k = int(self._cfg.actor_percentile * self._cfg.num_samples)
        proposal_k = int(self._cfg.proposal_percentile * self._cfg.num_samples)
        bigger_k = max(actor_k, proposal_k)

        _, top_idxs = jax.lax.top_k(q_vals, bigger_k)

        actor_update_actions = proposal_actions[top_idxs[:actor_k]]
        proposal_update_actions = proposal_actions[top_idxs[:proposal_k]]

        return UpdateActions(actor_update_actions, proposal_update_actions)

    def _policy_loss(self, params: chex.ArrayTree, policy: hk.Transformed, state: State, top_actions: jax.Array):
        out: PolicyOutputs = policy.apply(params=params, x=state.features.array)
        dist = distrax.MultivariateNormalDiag(out.mu, out.sigma)
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
        losses = jax_u.vmap(self._policy_loss, in_axes=(None, None, 0, 0))(params, policy, states, top_actions_batch)
        return jnp.mean(losses)
