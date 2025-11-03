from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax

import lib_agent.network.networks as nets
from lib_agent.buffer.datatypes import Transition
from lib_agent.critic.critic_protocol import CriticConfig
from lib_agent.critic.critic_utils import (
    CriticState,
    get_ensemble_norm,
    get_layer_norms,
    l2_regularizer,
)
from lib_agent.critic.rolling_reset import RollingResetManager


class AdvCriticOutputs(NamedTuple):
    q: jax.Array
    v: jax.Array
    h: jax.Array
    adv: jax.Array
    phi: jax.Array


class AdvNetworkOutputs(NamedTuple):
    adv: jax.Array
    phi: jax.Array


class AdvCriticMetrics(NamedTuple):
    v: jax.Array
    h: jax.Array
    adv: jax.Array
    loss: jax.Array
    v_loss: jax.Array
    h_loss: jax.Array
    adv_loss: jax.Array
    delta: jax.Array
    h_reg_loss: jax.Array
    ensemble_grad_norms: jax.Array
    ensemble_weight_norms: jax.Array
    layer_grad_norms: jax.Array
    layer_weight_norms: jax.Array


@dataclass
class AdvConfig(CriticConfig):
    polyak_tau: float = 0.0
    num_policy_actions: int = 100
    advantage_centering_weight: float = 0.1
    adv_l2_regularization: float = 1.0
    h_lr_mult: float = 1.0
    v_lr_mult: float = 1.0


def critic_builder(
    v_cfg: nets.TorsoConfig,
    h_cfg: nets.TorsoConfig,
    adv_cfg: nets.TorsoConfig,
):
    """Builds three independent critic networks: v, h, and adv."""

    def _v_network(x: jax.Array):
        v_torso = nets.torso_builder(v_cfg)
        phi_v = v_torso(x)
        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return hk.Linear(1, name='value_head', w_init=small_init, with_bias=True)(phi_v)

    def _h_network(x: jax.Array):
        h_torso = nets.torso_builder(h_cfg)
        phi_h = h_torso(x)
        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return hk.Linear(1, name='h_delta_head', w_init=small_init, with_bias=True)(phi_h)

    def _adv_network(x: jax.Array, a: jax.Array):
        adv_torso = nets.torso_builder(adv_cfg)
        # Concatenate state and action, then process through advantage torso
        sa = jnp.concatenate([x, a], axis=-1)
        phi_adv = adv_torso(sa)
        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return AdvNetworkOutputs(hk.Linear(1, name='adv_head', w_init=small_init, with_bias=False)(phi_adv), phi_adv)

    # Return separate transforms for each network
    return {
        'value': hk.transform(_v_network),
        'h_delta': hk.transform(_h_network),
        'adv': hk.transform(_adv_network),
    }


class AdvCritic:
    def __init__(self, cfg: AdvConfig, state_dim: int, action_dim: int):
        self._cfg = cfg
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._reset_manager = RollingResetManager(cfg.rolling_reset_config, cfg.ensemble)

        # Build three independent networks with separate torsos
        v_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=128, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=64, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=32, activation='crelu'),
                nets.LayerNormConfig(),
            ],
            skip=False,
        )

        h_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=128, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=64, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=32, activation='crelu'),
                nets.LayerNormConfig(),
            ],
            skip=False,
        )

        adv_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=128, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=64, activation='relu'),
                nets.LayerNormConfig(),
                nets.LinearConfig(size=32, activation='relu'),
                nets.LayerNormConfig(),
            ],
            skip=False,
        )

        self._nets = critic_builder(
            v_torso_cfg,
            h_torso_cfg,
            adv_torso_cfg,
        )

        def label(param_dict: dict, path_fn: Callable[[str], str]):
            def map_fn(nested_dict: dict | Any, path: str):
                if isinstance(nested_dict, dict):
                    return {k: map_fn(v, path + k) for k, v in nested_dict.items()}
                return path_fn(path)

            return map_fn(param_dict, '')

        def label_fn(path: str):
            if 'h_delta' in path:
                return 'h_delta'
            if 'value' in path:
                return 'value'
            return 'adv'

        params = self._init_params(jax.random.PRNGKey(0), jnp.ones(state_dim), jnp.ones(action_dim))

        self._optim = optax.partition(
            {
                'h_delta': optax.adamw(
                    learning_rate=cfg.stepsize * cfg.h_lr_mult,
                    weight_decay=0.001,
                ),
                'value': optax.adamw(
                    learning_rate=cfg.stepsize * cfg.v_lr_mult,
                    weight_decay=0.001,
                ),
                'adv': optax.adamw(
                    learning_rate=cfg.stepsize,
                    weight_decay=0.001,
                ),
            },
            param_labels=label(params, label_fn),
        )

    # ----------------------
    # -- Public Interface --
    # ----------------------
    @jax_u.method_jit
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        ens_init = jax_u.vmap_only(self._init_member_state, ['rng'])

        rngs = jax.random.split(rng, self._reset_manager.total_critics)
        return ens_init(rngs, x, a)

    def forward(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        only_active: bool = True,
    ):
        """Computes AdvCriticOutputs using ensemble members, optionally filtering to active members only."""
        chex.assert_rank(rng, 1)
        if only_active:
            active_indices = list(self._reset_manager.active_indices)
            active_indices.sort()
            active_indices = jnp.array(active_indices)
        else:
            active_indices = jnp.arange(self._reset_manager.total_critics)

        ens_get_values = jax_u.vmap_only(self._forward, ['params'])
        active_params = jax.tree.map(lambda x: x[active_indices], params)
        return ens_get_values(active_params, rng, state, action)

    def get_rolling_reset_metrics(self, prefix: str = "") -> dict[str, float]:
        metrics = {}
        for i in range(self._reset_manager.total_critics):
            critic_metrics = self._reset_manager.get_critic_metrics(i, prefix)
            metrics.update(critic_metrics)
        return metrics

    def get_representations(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        """Returns advantage network output as representation."""
        return self._forward_adv(params, rng, x, a)

    def update(
        self,
        seed: chex.PRNGKey,
        critic_state: CriticState,
        transitions: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        chex.assert_rank(seed, 1)
        update_rng, reset_rng = jax.random.split(seed)
        chex.assert_rank(transitions.state.features, 3)  # (ens, batch, state_dim)
        chex.assert_tree_shape_prefix(transitions, transitions.state.features.shape[:2])
        chex.assert_rank(policy_actions, 4)  # (ens, batch, samples, action_dim)
        chex.assert_rank(policy_probs, 3)  # (ens, batch, samples)

        self._reset_manager.increment_update_count()

        new_state, metrics = self._ensemble_update(
            critic_state,
            update_rng,
            transitions,
            policy_actions,
            policy_probs,
        )

        self._reset_manager.update_critic_metadata(metrics.loss)

        if self._reset_manager.should_reset():
            new_state = self._reset_manager.reset(
                new_state,
                reset_rng,
                self._init_member_state,
                self._state_dim,
                self._action_dim,
            )

        return new_state, metrics

    # -------------------------------
    # -- Shared net.apply vmapping --
    # -------------------------------
    @jax_u.method_jit
    def _forward_v(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
    ) -> jax.Array:
        # state shape is one of (state_dim,) or (batch, state_dim)
        f_v = self._nets['value'].apply

        if state.ndim == 1:
            # state (state_dim,)
            chex.assert_rank(rng, 1)
            f = f_v

        elif state.ndim == 2:
            # state (batch, state_dim)
            batch_size = state.shape[0]
            rng = jax.random.split(rng, batch_size)
            chex.assert_rank(rng, 2)
            f = jax_u.vmap(f_v, (None, 0, 0))

        else:
            raise ValueError(
                "Invalid state input shape. ",
                "State shape must be one of (state_dim,) or (batch, state_dim).",
            )

        return f(params['value'], rng, state)  # type: ignore

    @jax_u.method_jit
    def _forward_h(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
    ) -> jax.Array:
        # state shape is one of (state_dim,) or (batch, state_dim)
        f_h = self._nets['h_delta'].apply

        if state.ndim == 1:
            # state (state_dim,)
            chex.assert_rank(rng, 1)
            f = f_h

        elif state.ndim == 2:
            # state (batch, state_dim)
            batch_size = state.shape[0]
            rng = jax.random.split(rng, batch_size)
            chex.assert_rank(rng, 2)
            f = jax_u.vmap(f_h, (None, 0, 0))

        else:
            raise ValueError(
                "Invalid state input shape. ",
                "State shape must be one of (state_dim,) or (batch, state_dim).",
            )

        return f(params['h_delta'], rng, state)  # type: ignore

    @jax_u.method_jit
    def _forward_adv(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
    ) -> AdvNetworkOutputs:
        # state shape is one of (state_dim,) or (batch, state_dim)
        # if state is of shape (state_dim,), action must be of shape (action_dim,) or (n_samples, action_dim)
        # if state has batch dim, action must be of shape (batch, action_dim,) or (batch, n_samples, action_dim)
        f_adv = self._nets['adv'].apply

        if state.ndim == 1 and action.ndim == 1:
            # state (state_dim,) action (action_dim,)
            chex.assert_rank(rng, 1)
            f = f_adv

        elif state.ndim == 1 and action.ndim == 2:
            # state (state_dim,) action (n_samples, action_dim)
            n_samples = action.shape[0]
            rng = jax.random.split(rng, n_samples)
            chex.assert_rank(rng, 2)
            f = jax_u.vmap(f_adv, (None, 0, None, 0))

        elif state.ndim == 2 and action.ndim == 2:
            # state (batch, state_dim,) action (batch, action_dim
            chex.assert_equal_shape_prefix((state, action), prefix_len=1)
            batch_size = action.shape[0]
            rng = jax.random.split(rng, batch_size)
            chex.assert_rank(rng, 2)
            f = jax_u.vmap(f_adv, (None, 0, 0, 0))

        elif state.ndim == 2 and action.ndim == 3:
            # state (batch, state_dim,) action (batch, n_samples, action_dim)
            chex.assert_equal_shape_prefix((state, action), prefix_len=1)
            batch_size, n_samples = action.shape[:2]
            rng = jax.random.split(rng, (batch_size, n_samples))
            chex.assert_rank(rng, 3)
            f = jax_u.vmap(
                    jax_u.vmap(f_adv, (None, 0, None, 0)),  # inner maps over n_samples
                (None, 0, 0, 0),  # outer maps over batch
            )

        else:
            raise ValueError(
                "Invalid state and/or action input shapes. ",
                "State shape is one of (state_dim,) or (batch, state_dim). ",
                "If state is of shape (state_dim,), action must be of shape (action_dim,) or (n_samples, action_dim).",
                "If state has batch dim, ",
                "action must be of shape (batch, action_dim,) or (batch, n_samples, action_dim).",
            )

        return f(params['adv'], rng, state, action)  # type: ignore

    @jax_u.method_jit
    def _forward(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
    ) -> AdvCriticOutputs:
        """Forward pass through all three independent networks."""
        v_rng, h_rng, adv_rng = jax.random.split(rng, 3)

        # Call each network independently
        v = self._forward_v(params, v_rng, state)
        h = self._forward_h(params, h_rng, state)
        adv_out = self._forward_adv(params, adv_rng, state, action)
        chex.assert_equal_shape((v, h))

        if action.ndim == 3:  # if action is of shape (batch, n_samples, action_dim)
            # then we must reshape v and h to match by adding a dimension and repeats along that dim
            chex.assert_rank(state, 2)
            n_samples = action.shape[1]
            new_shape = (v.shape[0], n_samples, v.shape[1])
            v = jnp.broadcast_to(jnp.expand_dims(v, axis=1), new_shape)
            h = jnp.broadcast_to(jnp.expand_dims(h, axis=1), new_shape)
        elif action.ndim == 2 and state.ndim == 1: # if action is of shape (n_samples, action_dim)
            v = jnp.broadcast_to(jnp.expand_dims(v, axis=0), adv_out.adv.shape)
            h = jnp.broadcast_to(jnp.expand_dims(h, axis=0), adv_out.adv.shape)

        # Combine outputs
        q = v + adv_out.adv
        return AdvCriticOutputs(
            q=q,
            v=v,
            h=h,
            adv=adv_out.adv,
            phi=adv_out.phi,
        )

    # --------------------
    # -- Initialization --
    # --------------------
    def _init_params(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        v_rng, h_rng, adv_rng = jax.random.split(rng, 3)
        v_params = self._nets['value'].init(v_rng, x)
        h_params = self._nets['h_delta'].init(h_rng, x)
        adv_params = self._nets['adv'].init(adv_rng, x, a)
        return {
            'value': v_params,
            'h_delta': h_params,
            'adv': adv_params,
        }

    def _init_member_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        params = self._init_params(rng, x, a)
        return CriticState(
            params=params,
            opt_state=self._optim.init(params),
        )

    def initialize_to_nominal_action(
        self,
        rng: chex.PRNGKey,
        critic_state: CriticState,
        nominal_action: jax.Array,
    ):
        chex.assert_shape(nominal_action, (self._action_dim,))

        def regress_to_nominal(
            params: chex.ArrayTree,
            rng: chex.PRNGKey,
        ):
            BATCH = 32
            ACTION_SAMPLES = 128

            s_rng, a_rng, q_rng = jax.random.split(rng, 3)
            states = jax.random.uniform(s_rng, shape=(BATCH, self._state_dim))
            actions = jax.random.uniform(a_rng, shape=(BATCH, ACTION_SAMPLES, self._action_dim))
            adv = self._forward_adv(params, q_rng, states, actions).adv.squeeze()
            chex.assert_shape(adv, (BATCH, ACTION_SAMPLES))

            y = -jnp.abs(actions - jnp.expand_dims(nominal_action, axis=(0, 1))).sum(axis=-1)
            return jnp.square(adv - y).mean()

        @jax_u.jit
        @jax_u.vmap
        def update_params(
            params: chex.ArrayTree,
            opt_state: chex.ArrayTree,
            rng: chex.PRNGKey,
        ):
            loss, grad = jax.value_and_grad(regress_to_nominal)(params, rng)
            updates, new_opt_state = self._optim.update(
                grad,
                opt_state,
                params=params,
            )
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_opt_state

        params = critic_state.params
        opt_state = critic_state.opt_state
        for _ in range(self._cfg.nominal_setpoint_updates):
            rng, sub = jax.random.split(rng)
            ens_rng = jax.random.split(sub, self._reset_manager.total_critics)
            _, params, opt_state = update_params(params, opt_state, ens_rng)

        return critic_state._replace(
            params=params,
        )

    # ------------
    # -- Update --
    # ------------
    @jax_u.method_jit
    def _ensemble_update(
        self,
        state: CriticState,
        rng: chex.PRNGKey,
        transitions: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        """
        Updates each member of the ensemble.
        """
        chex.assert_rank(transitions.state.features, 3)  # (ens, batch, state_dim)
        chex.assert_tree_shape_prefix(transitions, transitions.state.features.shape[:2])
        chex.assert_rank(policy_actions, 4)  # (ens, batch, samples, action_dim)
        chex.assert_rank(policy_probs, 3)  # (ens, batch, samples)
        chex.assert_tree_shape_prefix(state, (self._reset_manager.total_critics,))

        grads, metrics = jax_u.grad(self._ensemble_loss, has_aux=True)(
            state.params,
            rng,
            transitions,
            policy_actions,
            policy_probs,
        )

        updates, new_opt_state = jax_u.vmap(self._optim.update, in_axes=0)(
            grads,
            state.opt_state,
            params=state.params,
        )
        updated_params = optax.apply_updates(state.params, updates)

        new_params = jax.tree.map(
            lambda old, updated: (1 - self._cfg.polyak_tau) * updated + self._cfg.polyak_tau * old,
            state.params,
            updated_params,
        )

        metrics = metrics._replace(
            ensemble_grad_norms=get_ensemble_norm(grads),
            ensemble_weight_norms=get_ensemble_norm(new_params),
            layer_grad_norms=get_layer_norms(grads),
            layer_weight_norms=get_layer_norms(new_params),
        )

        return CriticState(
            params=new_params,
            opt_state=new_opt_state,
        ), metrics

    def _ensemble_loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        chex.assert_rank(transition.state.features, 3)  # (ens, batch, state_dim)
        chex.assert_tree_shape_prefix(transition, transition.state.features.shape[:2])
        chex.assert_rank(policy_actions, 4)  # (ens, batch, samples, action_dim)
        chex.assert_rank(policy_probs, 3)  # (ens, batch, samples)
        rngs = jax.random.split(rng, self._reset_manager.total_critics)
        losses, metrics = jax_u.vmap(self._batch_loss)(
            params,
            rngs,
            transition,
            policy_actions,
            policy_probs,
        )

        return losses.sum(), metrics

    def _batch_loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        chex.assert_rank(policy_actions, 3)  # (batch, samples, action_dim)
        chex.assert_rank(policy_probs, 2)  # (batch, samples)
        chex.assert_rank(transition.state.features, 2)  # (batch, state_dim)
        chex.assert_rank(rng, 1)
        rngs = jax.random.split(rng, policy_actions.shape[0])
        chex.assert_shape(rngs, (policy_actions.shape[0], 2))
        losses, metrics = jax_u.vmap_except(self._loss, ['params'])(
            params,
            rngs,
            transition,
            policy_actions,
            policy_probs,
        )
        # Compute regularization for h and adv networks separately
        h_reg_loss = l2_regularizer(params['h_delta']['h_delta_head'], self._cfg.l2_regularization)  # type: ignore
        adv_reg_loss = l2_regularizer(params['adv']['adv_head'], self._cfg.adv_l2_regularization)  # type: ignore

        metrics = metrics._replace(h_reg_loss=h_reg_loss)
        return losses.mean() + h_reg_loss + adv_reg_loss, metrics

    def _loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        state = transition.state
        action = transition.action
        n_step_reward = transition.n_step_reward
        next_state = transition.next_state
        n_step_gamma = transition.n_step_gamma
        chex.assert_rank((rng, state.features, next_state.features, action), 1)
        chex.assert_rank(policy_actions, 2)  # (num_samples, action_dim)
        chex.assert_rank(policy_probs, 1)  # (num_samples,)
        chex.assert_rank((n_step_reward, n_step_gamma), 0)  # scalars

        v_rng, h_rng, adv_rng, vp_rng, centering_rng, rand_a_rng = jax.random.split(rng, 6)

        v = self._forward_v(params, v_rng, state.features.array)
        h = self._forward_h(params, h_rng, state.features.array)
        adv = self._forward_adv(params, adv_rng, state.features.array, action).adv
        v_prime = self._forward_v(params, vp_rng, next_state.features.array)

        target = n_step_reward + n_step_gamma * v_prime

        sg = jax.lax.stop_gradient
        delta = sg(target) - v

        v_loss = 0.5 * delta**2 + sg(jnp.tanh(h)) * delta
        h_loss = 0.5 * (sg(delta) - h)**2

        # loss for advantage function
        adv_loss = 0.5 * ((sg(delta) - sg(h)) - adv)**2

        # Advantage centering loss: lambda * (sum(pi(a,s) * A(a,s)))^2
        # Compute advantages for all policy actions
        num_samples = policy_probs.shape[0]
        policy_advantages = self._forward_adv(params, centering_rng, state.features.array, policy_actions).adv.squeeze()
        chex.assert_shape(policy_advantages, (num_samples,))

        # Weighted sum of advantages by policy probabilities
        weighted_adv_sum = jnp.mean(policy_probs * policy_advantages)
        centering_loss = self._cfg.advantage_centering_weight * 0.5 * jnp.square(weighted_adv_sum)

        # optimism loss
        unif_rng, rand_a_rng = jax.random.split(rand_a_rng, 2)
        rand_actions = jax.random.uniform(unif_rng, shape=(self._cfg.num_rand_actions, self._action_dim))
        rand_actions_in_bounds = ((rand_actions + transition.state.a_lo)
                                  * (transition.state.a_hi - transition.state.a_lo))

        rand_advs = self._forward_adv(params, rand_a_rng, state.features.array, rand_actions_in_bounds).adv.squeeze()
        action_reg_loss = self._cfg.action_regularization * 0.5 * jnp.square(rand_advs.mean())
        loss = v_loss + h_loss + adv_loss + centering_loss + action_reg_loss

        metrics = AdvCriticMetrics(
            v=v.squeeze(),
            h=h.squeeze(),
            adv=adv.squeeze(),
            loss=loss.squeeze(),
            v_loss=v_loss.squeeze(),
            h_loss=h_loss.squeeze(),
            adv_loss=adv_loss.squeeze(),
            delta=delta.squeeze(),

            # filled out further up
            h_reg_loss=jnp.array(0),
            ensemble_grad_norms=jnp.array(0),
            ensemble_weight_norms=jnp.array(0),
            layer_grad_norms=jnp.array(0),
            layer_weight_norms=jnp.array(0),
        )

        return loss, metrics
