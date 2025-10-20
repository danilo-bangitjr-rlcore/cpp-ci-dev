from dataclasses import dataclass
from typing import NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax

import lib_agent.network.networks as nets
from lib_agent.critic.critic_protocol import CriticConfig
from lib_agent.critic.critic_utils import (
    CriticBatch,
    CriticState,
    get_ensemble_norm,
    get_layer_norms,
    l2_regularizer,
)
from lib_agent.critic.rolling_reset import RollingResetManager


class AdvParams(NamedTuple):
    adv_params: chex.ArrayTree
    val_params: chex.ArrayTree


class ValueCriticOutputs(NamedTuple):
    v: jax.Array
    h: jax.Array
    phi: jax.Array


class AdvCriticOutputs(NamedTuple):
    q: jax.Array
    v: jax.Array
    h: jax.Array
    adv: jax.Array
    phi_state: jax.Array
    phi_joint: jax.Array


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
    num_policy_actions: int = 100
    advantage_centering_weight: float = 0.1
    adv_l2_regularization: float = 1.0


def critic_builder(
    state_cfg: nets.TorsoConfig,
    action_cfg: nets.TorsoConfig,
    joint_cfg: nets.TorsoConfig,
):
    """Builds unified critic network with separate state and action processing."""

    def _compute_state_features(x: jax.Array):
        """Compute state features and value (no action needed)."""
        # Non-linear state processing
        state_torso = nets.torso_builder(state_cfg)
        phi_state = state_torso(x)

        small_init = hk.initializers.VarianceScaling(scale=0.0001)

        v = hk.Linear(1, name='v_nonlinear', w_init=small_init, with_bias=True)(phi_state)
        h = hk.Linear(1, name='h', w_init=small_init, with_bias=True)(phi_state)

        return v, h, phi_state

    def _compute_advantage(phi_state: jax.Array, a: jax.Array):
        """Compute advantage given state features and action."""
        action_torso = nets.torso_builder(action_cfg)
        phi_action = action_torso(a)

        phi_joint = jnp.concatenate([phi_state, phi_action], axis=-1)
        joint_torso = nets.torso_builder(joint_cfg)
        phi_joint = joint_torso(phi_joint)

        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        adv = hk.Linear(1, name='adv', w_init=small_init, with_bias=False)(phi_joint)

        return adv, phi_joint

    def _full(x: jax.Array, a: jax.Array):
        """Compute all outputs."""
        v, h, phi_state = _compute_state_features(x)
        adv, phi_joint = _compute_advantage(phi_state, a)
        return AdvCriticOutputs(q=v + adv, v=v, h=h, adv=adv, phi_state=phi_state, phi_joint=phi_joint)

    def _value_only(x: jax.Array):
        """Compute only value outputs."""
        v, h, phi_state = _compute_state_features(x)
        return ValueCriticOutputs(v=v, h=h, phi=phi_state)

    # Return transforms for different use cases
    return {
        'full': hk.transform(_full),
        'value': hk.transform(_value_only),
    }


class AdvCritic:
    def __init__(self, cfg: AdvConfig, seed: int, state_dim: int, action_dim: int):
        self._rng = jax.random.PRNGKey(seed)
        self._cfg = cfg
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._reset_manager = RollingResetManager(cfg.rolling_reset_config, cfg.ensemble)

        layer_norm_cfg = (
            nets.LayerNormConfig
            if cfg.use_all_layer_norm
            else nets.IdentityConfig
        )

        # build unified network with separate state and action streams
        state_torso_cfg = nets.TorsoConfig(
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

        action_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=32, activation='relu'),
                layer_norm_cfg(),
                nets.LinearConfig(size=32, activation='crelu'),
                layer_norm_cfg(),
            ],
            skip=False,
        )

        joint_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=64, activation='relu'),
                layer_norm_cfg(),
                nets.LinearConfig(size=64, activation='relu'),
                layer_norm_cfg(),
            ],
            skip=False,
        )

        self._unified_net = critic_builder(
            state_torso_cfg,
            action_torso_cfg,
            joint_torso_cfg,
        )

        self._optim = optax.adamw(learning_rate=cfg.stepsize, weight_decay=0.001)

    # ----------------------
    # -- Public Interface --
    # ----------------------
    @jax_u.method_jit
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        ens_init = jax_u.vmap_only(self._init_member_state, ['rng'])

        rngs = jax.random.split(rng, self._reset_manager.total_critics)
        return ens_init(rngs, x, a)

    def get_values(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array, action: jax.Array):
        return self._forward(params, rng, state, action)

    def get_active_indices(self):
        indices = self._reset_manager.active_indices
        return jnp.array(sorted(indices))

    def get_rolling_reset_metrics(self, prefix: str = "") -> dict[str, float]:
        metrics = {}
        for i in range(self._reset_manager.total_critics):
            critic_metrics = self._reset_manager.get_critic_metrics(i, prefix)
            metrics.update(critic_metrics)
        return metrics

    def get_active_values(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
    ):
        active_indices = self.get_active_indices()
        return self._get_active_values(params, rng, state, action, active_indices)

    @jax_u.method_jit
    def _get_active_values(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        active_indices: jax.Array,
    ):
        ens_get_values = jax_u.vmap_only(self.get_values, ['params'])
        active_params = jax.tree.map(lambda x: x[active_indices], params)
        return ens_get_values(active_params, rng, state, action)

    def get_representations(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        return self._forward(params, rng, x, a).phi_joint

    def update(
        self,
        critic_state: CriticState,
        transitions: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        self._rng, update_rng, reset_rng = jax.random.split(self._rng, 3)
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
    def _forward(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
    ) -> AdvCriticOutputs:
        # state shape is one of (state_dim,) or (batch, state_dim)
        # if state is of shape (state_dim,), action must be of shape (action_dim,) or (n_samples, action_dim)
        # if state has batch dim, action must be of shape (batch, action_dim,) or (batch, n_samples, action_dim)
        f = self._unified_net['full'].apply
        if action.ndim == state.ndim + 1:
            # vmap over action samples and rngs
            f = jax_u.vmap(f, (None, 0, None, 0))

        if state.ndim == 1:
            return f(params, rng, state, action)

        # batch mode - vmap over batch dim
        chex.assert_rank(state, 2)
        f = jax_u.vmap(f, (None, 0, 0, 0))
        return f(params, rng, state, action)

    @jax_u.method_jit
    def _forward_val(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array) -> ValueCriticOutputs:
        # state shape is one of (state_dim,) or (batch, state_dim)
        # Use value-only transform - no action processing!
        f = self._unified_net['value'].apply

        if state.ndim == 1:
            return f(params, rng, state)

        # batch mode - vmap over batch dim
        chex.assert_rank(state, 2)
        rngs = jax.random.split(rng, state.shape[0])
        f = jax_u.vmap(f, (None, 0, 0))
        return f(params, rngs, state)

    # --------------------
    # -- Initialization --
    # --------------------
    def _init_member_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        # Initialize using the full network to create all parameters
        params = self._unified_net['full'].init(rng, x, a)

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
            q_rngs = jax.random.split(q_rng, (BATCH, ACTION_SAMPLES))
            states = jax.random.uniform(s_rng, shape=(BATCH, self._state_dim))
            actions = jax.random.uniform(a_rng, shape=(BATCH, ACTION_SAMPLES, self._action_dim))
            adv = self._forward(params, q_rngs, states, actions).adv.squeeze()
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
        transitions: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        """
        Updates each member of the ensemble.
        """
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
        new_params = optax.apply_updates(state.params, updates)

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
        transition: CriticBatch,
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
        transition: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        chex.assert_rank(policy_actions, 3)  # (batch, samples, action_dim)
        chex.assert_rank(policy_probs, 2)  # (batch, samples)
        batch_size = policy_actions.shape[0]
        rngs = jax.random.split(rng, batch_size)
        losses, metrics = jax_u.vmap_except(self._loss, ['params'])(
            params,
            rngs,
            transition,
            policy_actions,
            policy_probs,
        )
        h_reg_loss = l2_regularizer(params['h'], self._cfg.l2_regularization)  # type: ignore
        adv_reg_loss = l2_regularizer(params['adv'], self._cfg.adv_l2_regularization)  # type: ignore

        metrics = metrics._replace(h_reg_loss=h_reg_loss)
        return losses.mean() + h_reg_loss + adv_reg_loss, metrics

    def _loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma
        chex.assert_rank((state.features, next_state.features, action), 1)
        chex.assert_rank(policy_actions, 2)  # (num_samples, action_dim)
        chex.assert_rank(policy_probs, 1)  # (num_samples,)
        chex.assert_rank((reward, gamma), 0)  # scalars

        v_rng, vp_rng, centering_rng, rand_a_rng, opt_rng = jax.random.split(rng, 5)

        out = self._forward(params, v_rng, state.features.array, action)
        v = out.v
        h = out.h
        adv = out.adv

        # v_prime is the state value of the next state
        v_prime = self._forward_val(params, vp_rng, next_state.features.array).v
        target = reward + gamma * v_prime

        sg = jax.lax.stop_gradient
        delta = sg(target) - v

        v_loss = 0.5 * delta**2 + sg(jnp.tanh(h)) * delta
        h_loss = 0.5 * (sg(delta) - h)**2

        # loss for advantage function
        adv_loss = 0.5 * ((sg(delta) - sg(h)) - adv)**2

        # Advantage centering loss: lambda * (sum(pi(a,s) * A(a,s)))^2
        # Compute advantages for all policy actions
        num_samples = policy_probs.shape[0]
        centering_rngs = jax.random.split(centering_rng, num_samples)
        policy_advantages = self._forward(params, centering_rngs, state.features.array, policy_actions).adv.squeeze()
        chex.assert_shape(policy_advantages, (num_samples,))

        # Weighted sum of advantages by policy probabilities
        weighted_adv_sum = jnp.mean(policy_probs * policy_advantages)
        centering_loss = self._cfg.advantage_centering_weight * 0.5 * jnp.square(weighted_adv_sum)

        # optimism loss
        rand_actions = jax.random.uniform(rand_a_rng, shape=(self._cfg.num_rand_actions, self._action_dim))
        rand_actions_in_bounds = ((rand_actions + transition.state.a_lo)
                                  * (transition.state.a_hi - transition.state.a_lo))

        opt_rngs = jax.random.split(opt_rng, self._cfg.num_rand_actions)
        rand_advs = self._forward(params, opt_rngs, state.features.array, rand_actions_in_bounds).adv.squeeze()

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
