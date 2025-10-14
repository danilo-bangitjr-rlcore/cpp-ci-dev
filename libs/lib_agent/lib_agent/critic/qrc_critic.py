from dataclasses import dataclass, field
from typing import Any, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax

import lib_agent.network.networks as nets
from lib_agent.critic.critic_utils import (
    CriticBatch,
    CriticState,
    QRCCriticMetrics,
    RollingResetConfig,
    get_ensemble_norm,
    get_layer_norms,
    l2_regularizer,
    uniform_except,
)
from lib_agent.critic.rolling_reset import RollingResetManager


class QRCOutputs(NamedTuple):
    q: jax.Array
    h: jax.Array
    phi: jax.Array


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return QRCOutputs(
            q=hk.Linear(1, w_init=small_init, with_bias=False)(phi),
            h=hk.Linear(1, name='h', w_init=small_init, with_bias=False)(phi),
            phi=phi,
        )

    return hk.transform(_inner)


@dataclass
class QRCConfig:
    name: str
    stepsize: float
    ensemble: int
    ensemble_prob: float
    num_rand_actions: int
    action_regularization: float
    action_regularization_epsilon: float
    l2_regularization: float
    nominal_setpoint_updates: int = 1000
    use_all_layer_norm: bool = False
    rolling_reset_config: RollingResetConfig = field(default_factory=RollingResetConfig)


class QRCCritic:
    def __init__(self, cfg: QRCConfig, seed: int, state_dim: int, action_dim: int):
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

        torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(
                    streams=[
                        # states
                        [
                            nets.LinearConfig(size=128, activation='relu'),
                            nets.LayerNormConfig(),
                            nets.LinearConfig(size=64, activation='relu'),
                            nets.LayerNormConfig(),
                            nets.LinearConfig(size=32, activation='crelu'),
                            nets.LayerNormConfig(),
                        ],
                        # actions
                        [
                            nets.LinearConfig(size=32, activation='relu'),
                            layer_norm_cfg(),
                            nets.LinearConfig(size=32, activation='crelu'),
                            layer_norm_cfg(),
                        ],
                    ],
                ),
                nets.LinearConfig(size=64, activation='relu'),
                layer_norm_cfg(),
                nets.LinearConfig(size=64, activation='relu'),
                layer_norm_cfg(),
            ],
            skip=False,
        )
        self._net = critic_builder(torso_cfg)
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
        return self._forward(params, rng, x, a).phi

    def update(self, critic_state: Any, transitions: CriticBatch, next_actions: jax.Array):
        self._rng, update_rng, reset_rng = jax.random.split(self._rng, 3)
        self._reset_manager.increment_update_count()

        new_state, metrics = self._ensemble_update(
            critic_state,
            update_rng,
            transitions,
            next_actions,
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
    def _forward(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array, action: jax.Array) -> QRCOutputs:
        # state shape is one of (state_dim,) or (batch, state_dim)
        # if state is of shape (state_dim,), action must be of shape (action_dim,) or (n_samples, action_dim)
        # if state has batch dim, action must be of shape (batch, action_dim,) or (batch, n_samples, action_dim)
        f = self._net.apply
        if action.ndim == state.ndim + 1:
            # vmap over action samples and rngs
            f = jax_u.vmap(f, (None, 0, None, 0))

        if state.ndim == 1:
            return f(params, rng, state, action)

        # batch mode - vmap over batch dim
        chex.assert_rank(state, 2)
        f = jax_u.vmap(f, (None, 0, 0, 0))
        return f(params, rng, state, action)

    # --------------------
    # -- Initialization --
    # --------------------
    def _init_member_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        params = self._net.init(rng, x, a)
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
            q = self._forward(params, q_rngs, states, actions).q.squeeze()
            chex.assert_shape(q, (BATCH, ACTION_SAMPLES))

            y = -jnp.abs(actions - jnp.expand_dims(nominal_action, axis=(0, 1))).sum(axis=-1)
            return jnp.square(q - y).mean()

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
        next_actions: jax.Array,
    ):
        """
        Updates each member of the ensemble.
        """
        grads, metrics = jax_u.grad(self._ensemble_loss, has_aux=True)(
            state.params,
            rng,
            transitions,
            next_actions,
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
        next_actions: jax.Array,
    ):
        chex.assert_rank(transition.state.features, 3)  # (ens, batch, state_dim)
        chex.assert_tree_shape_prefix(transition, transition.state.features.shape[:2])
        rngs = jax.random.split(rng, self._reset_manager.total_critics)
        losses, metrics = jax_u.vmap(self._batch_loss)(
            params,
            rngs,
            transition,
            next_actions,
        )

        return losses.sum(), metrics

    def _batch_loss(
        self,
        params: Any,
        rng: chex.PRNGKey,
        transition: CriticBatch,
        next_actions: jax.Array,
    ):
        # (batch, samples, action_dim)
        chex.assert_rank(next_actions, 3)
        rngs = jax.random.split(rng, next_actions.shape[0])
        losses, metrics = jax_u.vmap_only(self._loss, ['rng', 'transition', 'next_actions'])(
            params,
            rngs,
            transition,
            next_actions,
        )
        h_reg_loss = l2_regularizer(params['h'], self._cfg.l2_regularization)
        metrics = metrics._replace(h_reg_loss=h_reg_loss)
        return losses.mean() + h_reg_loss, metrics

    def _loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: CriticBatch,
        next_actions: jax.Array,
    ):
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma
        chex.assert_rank((state.features, next_state.features, action), 1)
        chex.assert_rank(next_actions, 2)  # (num_samples, action_dim)
        chex.assert_rank((reward, gamma), 0)  # scalars

        q_rng, qp_rng, a_rng = jax.random.split(rng, 3)
        qp_rngs = jax.random.split(qp_rng, self._cfg.num_rand_actions)

        out = self._forward(params, q_rng, state.features, action)
        q = out.q
        h = out.h

        # q_prime takes expectation of state-action value over actions sampled from some dist
        q_prime = self.get_values(params, qp_rngs, next_state.features, next_actions).q.mean()

        target = reward + gamma * q_prime

        sg = jax.lax.stop_gradient
        delta_l = sg(target) - q
        delta_r = target - sg(q)

        q_loss = 0.5 * delta_l**2 + sg(jnp.tanh(h)) * delta_r
        h_loss = 0.5 * (sg(delta_l) - h)**2

        # optimism loss
        rand_actions = uniform_except(
            a_rng,
            shape=(self._cfg.num_rand_actions, action.shape[0]),
            val=action,
            epsilon=self._cfg.action_regularization_epsilon * (state.a_hi - state.a_lo),
            minval=state.a_lo,
            maxval=state.a_hi,
        )
        out_rand = jax_u.vmap_only(self._net.apply, [1, 3])(params, qp_rngs, state.features, rand_actions)
        action_reg_loss = self._cfg.action_regularization * jnp.abs(out_rand.q).mean()

        loss = q_loss + h_loss + action_reg_loss

        metrics = QRCCriticMetrics(
            q=q,
            h=h,
            loss=loss,
            q_loss=q_loss,
            h_loss=h_loss,
            delta_l=delta_l,
            delta_r=delta_r,
            action_reg_loss=action_reg_loss,

            # filled out further up
            h_reg_loss=jnp.array(0),
            ensemble_grad_norms=jnp.array(0),
            ensemble_weight_norms=jnp.array(0),
            layer_grad_norms=jnp.array(0),
            layer_weight_norms=jnp.array(0),
        )

        return loss, metrics
