from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax
from ml_instrumentation.Collector import Collector

import lib_agent.network.networks as nets
from lib_agent.buffer.buffer import VectorizedTransition


class ActionSampler(Protocol):
    def __call__(self, key: chex.PRNGKey, state: chex.Array) -> chex.Array: ...


class CriticState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree


class CriticOutputs(NamedTuple):
    q: jax.Array
    h: jax.Array


@dataclass
class QRCConfig:
    name: str
    stepsize: float
    ensemble: int
    ensemble_prob: float
    batch_size: int


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return CriticOutputs(
            q=hk.Linear(1)(phi),
            h=hk.Linear(1, name='h', w_init=small_init)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))

class QRCCritic:
    def __init__(self, cfg: QRCConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self._rng = jax.random.PRNGKey(seed)
        self._cfg = cfg
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._collector = collector

        torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(sizes=[123, 123], activation='relu'),
                nets.LinearConfig(size=225, activation='relu'),
            ],
        )
        self._net = critic_builder(torso_cfg)
        self._optim = optax.adam(learning_rate=cfg.stepsize)

    # ----------------------
    # -- Public Interface --
    # ----------------------
    @jax_u.method_jit
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        ens_init = jax_u.vmap_only(self._init_member_state, ['rng'])

        rngs = jax.random.split(rng, self._cfg.ensemble)
        return ens_init(rngs, x, a)

    def _init_member_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        params = self._net.init(rng, x, a)
        return CriticState(
            params=params,
            opt_state=self._optim.init(params),
        )

    @jax_u.method_jit
    def forward(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array):
        # ensemble mode
        if x.ndim == 3 and a.ndim == 3:
            chex.assert_equal_shape_prefix((x, a), 2)
            ens_forward = jax.vmap(self._forward)
            return ens_forward(params, x, a)

        # batch mode
        return self._forward(params, x, a)

    def _forward(self, params: chex.ArrayTree, state: jax.Array, action: jax.Array):
        return self._net.apply(params, state, action).q


    def update(self, critic_state: Any, transitions: VectorizedTransition, next_actions: jax.Array):
        new_state, metrics = self._ensemble_update(
            critic_state,
            transitions,
            next_actions,
        )

        loss = jnp.mean(metrics['losses'])
        self._collector.collect('critic_loss', float(loss))

        return new_state, metrics['losses']

    # ------------
    # -- Update --
    # ------------
    @jax_u.method_jit
    def _ensemble_update(
        self,
        state: CriticState,
        transitions: VectorizedTransition,
        next_actions: jax.Array,
    ):
        """
        Updates each member of the ensemble.
        """
        grads, metrics = jax.grad(self._ensemble_loss, has_aux=True)(
            state.params,
            transitions,
            next_actions,
        )

        ens_updates = []
        ens_opts = []
        for i in range(self._cfg.ensemble):
            updates, new_opt_state = self._optim.update(
                get_member(grads, i),
                get_member(state.opt_state, i),
                get_member(state.params, i),
            )

            ens_updates.append(updates)
            ens_opts.append(new_opt_state)

        updates = jax.tree_util.tree_map(lambda *upd: jnp.stack(upd, axis=0), *ens_updates)
        new_opt_state = jax.tree_util.tree_map(lambda *opt: jnp.stack(opt, axis=0), *ens_opts)
        new_params = optax.apply_updates(state.params, updates)

        return CriticState(
            new_params,
            new_opt_state,
        ), metrics


    def _ensemble_loss(
        self,
        params: chex.ArrayTree,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ):
        losses, h_losses, metrics = jax.vmap(self._batch_loss)(
            params,
            transition,
            next_actions,
        )
        return losses.sum() + h_losses.sum(), metrics | {
            'losses': losses,
            'h_losses': h_losses,
        }


    def _batch_loss(
        self,
        params: Any,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ):
        losses, h_losses, metrics = jax_u.vmap_only(self._loss, ['transition', 'next_actions'])(
            params,
            transition,
            next_actions,
        )
        return losses.mean(), h_losses.mean() + l2_regularizer(params['h'], 2.0), metrics


    def _loss(
        self,
        params: chex.ArrayTree,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ):
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma

        out = self._net.apply(params, state, action)
        out_p = self._net.apply(params, next_state, next_actions)

        target = reward + gamma * out_p.q.mean()

        sg = jax.lax.stop_gradient
        delta_l = sg(target) - out.q
        delta_r = target - sg(out.q)

        q_loss = 0.5 * delta_l**2 + sg(jnp.tanh(out.h)) * delta_r
        h_loss = 0.5 * (sg(delta_l) - out.h)**2
        return q_loss, h_loss, {
            'q': out.q,
            'h': out.h,
            'delta': delta_l,
        }

def get_member(a: chex.ArrayTree, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], a)


tree_map = jax.tree_util.tree_map
def l2_regularizer(params: chex.ArrayTree, beta: float):
    reg = tree_map(jnp.square, params)
    reg = tree_map(jnp.sum, reg)
    reg = 0.5 * beta * jax.tree_util.tree_reduce(lambda a, b: a + b, reg)
    return reg
