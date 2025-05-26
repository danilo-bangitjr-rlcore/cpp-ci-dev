from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_instrumentation.Collector import Collector

import agent.components.networks.networks as nets
import utils.jax as jax_u
from agent.components.buffer import EnsembleReplayBuffer, VectorizedTransition
from interaction.transition_creator import Transition


class CriticState(NamedTuple):
    params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: chex.ArrayTree


class CriticOutputs(NamedTuple):
    q: jax.Array


@dataclass
class SARSAConfig:
    stepsize: float
    ensemble: int
    ensemble_prob: float
    batch_size: int

    polyak: float


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        return CriticOutputs(
            q=hk.Linear(1)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))

class SARSACritic:
    def __init__(self, cfg: SARSAConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self._rng = jax.random.PRNGKey(seed)
        self._cfg = cfg
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._collector = collector

        torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(sizes=[128, 128], activation='relu'),
                nets.LinearConfig(size=256, activation='relu'),
            ],
        )
        self._net = critic_builder(torso_cfg)

        self._optim = optax.chain(
            optax.adam(learning_rate=cfg.stepsize),
            optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=50,
                max_learning_rate=cfg.stepsize,
                decrease_factor=0.9,
                increase_factor=np.inf,
                slope_rtol=0.1,
            ),
        )

        self._buffer = EnsembleReplayBuffer(
            n_ensemble=cfg.ensemble,
            ensemble_prob=cfg.ensemble_prob,
            batch_size=cfg.batch_size,
        )

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
            target_params=params,
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


    def update(self, state: Any, get_actions: Callable[[chex.PRNGKey, jax.Array], jax.Array]) -> CriticState:
        if self._buffer.size == 0:
            return state

        transitions = self._buffer.sample()
        self._rng, rng = jax.random.split(self._rng, 2)
        next_actions = get_actions(rng, transitions.next_state)

        new_state, losses = self._ensemble_update(
            state,
            transitions,
            next_actions,
        )

        loss = jnp.mean(losses)
        self._collector.collect('critic_loss', float(loss))

        return new_state

    def update_buffer(self, transition: Transition):
        self._buffer.add(transition)

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
        grads, member_losses = jax.grad(self._ensemble_loss, has_aux=True)(
            state.params,
            state.target_params,
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
                value=member_losses[i],
                grad=get_member(grads, i),
                value_fn=self._batch_loss,
                ens_target_params=state.target_params,
                transition=get_member(transitions, i),
                next_actions=get_member(next_actions, i),
            )

            ens_updates.append(updates)
            ens_opts.append(new_opt_state)

        updates = jax.tree_util.tree_map(lambda *upd: jnp.stack(upd, axis=0), *ens_updates)
        new_opt_state = jax.tree_util.tree_map(lambda *opt: jnp.stack(opt, axis=0), *ens_opts)
        new_params = optax.apply_updates(state.params, updates)

        # Target Net Polyak Update
        target_params = state.target_params
        new_target_params = optax.incremental_update(new_params, target_params, 1 - self._cfg.polyak)

        return CriticState(
            new_params,
            new_target_params,
            new_opt_state,
        ), member_losses


    def _ensemble_loss(
        self,
        params: chex.ArrayTree,
        ens_target_params: chex.ArrayTree,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ):
        losses = jax_u.vmap_except(self._batch_loss, ['ens_target_params'])(
            params,
            ens_target_params,
            transition,
            next_actions,
        )
        return losses.sum(), losses


    def _batch_loss(
        self,
        params: chex.ArrayTree,
        ens_target_params: chex.ArrayTree,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ):
        losses = jax_u.vmap_only(self._loss, ['transition', 'next_actions'])(
            params,
            ens_target_params,
            transition,
            next_actions,
        )
        return losses.mean()


    def _loss(
        self,
        params: chex.ArrayTree,
        ens_target_params: chex.ArrayTree,
        transition: VectorizedTransition,
        next_actions: jax.Array,
    ) -> jax.Array:
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        gamma = transition.gamma

        q = self.forward(params, state, action)
        qp = jax_u.vmap_only(self._forward, ['params'])(ens_target_params, next_state, next_actions).mean(axis=0)
        target = reward + gamma * qp
        loss = 0.5 * (target - q)**2
        return loss

def get_member(a: chex.ArrayTree, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], a)
