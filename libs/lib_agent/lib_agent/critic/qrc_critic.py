from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Protocol

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import optax
from lib_utils import dict as dict_u

import lib_agent.network.networks as nets
from lib_agent.buffer.buffer import State


class ActionSampler(Protocol):
    def __call__(self, key: chex.PRNGKey, state: chex.Array) -> chex.Array: ...


class CriticState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree


class CriticOutputs(NamedTuple):
    q: jax.Array
    h: jax.Array
    phi: jax.Array


class CriticBatch(Protocol):
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
    use_noisy_nets: bool = False


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        small_init = hk.initializers.VarianceScaling(scale=0.0001)
        return CriticOutputs(
            q=hk.Linear(1, w_init=small_init, with_bias=False)(phi),
            h=hk.Linear(1, name='h', w_init=small_init, with_bias=False)(phi),
            phi=phi,
        )

    return hk.transform(_inner)

class QRCCritic:
    def __init__(self, cfg: QRCConfig, seed: int, state_dim: int, action_dim: int):
        self._rng = jax.random.PRNGKey(seed)
        self._cfg = cfg
        self._state_dim = state_dim
        self._action_dim = action_dim

        interior_layer_cfg = (
            nets.NoisyLinearConfig
            if cfg.use_noisy_nets
            else nets.LinearConfig
        )

        torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(
                    streams=[
                        # states
                        [
                            interior_layer_cfg(size=128, activation='relu'),
                            interior_layer_cfg(size=64, activation='relu'),
                            interior_layer_cfg(size=32, activation='crelu'),
                        ],
                        # actions
                        [
                            interior_layer_cfg(size=32, activation='relu'),
                            interior_layer_cfg(size=32, activation='crelu'),
                        ],
                    ],
                ),
                interior_layer_cfg(size=64, activation='relu'),
                interior_layer_cfg(size=64, activation='relu'),
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

        rngs = jax.random.split(rng, self._cfg.ensemble)
        return ens_init(rngs, x, a)

    def get_values(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array, action: jax.Array):
        return self._forward(params, rng, state, action).q

    def get_representations(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        return self._forward(params, rng, x, a).phi

    def update(self, critic_state: Any, transitions: CriticBatch, next_actions: jax.Array):
        self._rng, rng = jax.random.split(self._rng)
        new_state, metrics = self._ensemble_update(
            critic_state,
            rng,
            transitions,
            next_actions,
        )
        return new_state, metrics

    # -------------------------------
    # -- Shared net.apply vmapping --
    # -------------------------------
    @jax_u.method_jit
    def _forward(self, params: chex.ArrayTree, rng: chex.PRNGKey, state: jax.Array, action: jax.Array) -> CriticOutputs:
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
            ens_rng = jax.random.split(sub, self._cfg.ensemble)
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
            new_params,
            new_opt_state,
        ), metrics


    def _ensemble_loss(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        transition: CriticBatch,
        next_actions: jax.Array,
    ):
        chex.assert_rank(transition.state.features, 3) # (ens, batch, state_dim)
        chex.assert_tree_shape_prefix(transition, transition.state.features.shape[:2])
        rngs = jax.random.split(rng, self._cfg.ensemble)
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
        chex.assert_rank(next_actions, 2) # (num_samples, action_dim)
        chex.assert_rank((reward, gamma), 0) # scalars

        q_rng, qp_rng, a_rng = jax.random.split(rng, 3)
        qp_rngs = jax.random.split(qp_rng, self._cfg.num_rand_actions)

        out = self._forward(params, q_rng, state.features, action)
        q = out.q
        h = out.h

        # q_prime takes expectation of state-action value over actions sampled from some dist
        q_prime = self.get_values(params, qp_rngs, next_state.features, next_actions).mean()

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


def l2_regularizer(params: chex.ArrayTree, beta: float):
    reg = jax.tree.map(jnp.square, params)
    reg = jax.tree.map(jnp.sum, reg)
    return 0.5 * beta * jax.tree.reduce(lambda a, b: a + b, reg)

# ---------------------------------------------------------------------------- #
#                                    metrics                                   #
# ---------------------------------------------------------------------------- #
class QRCCriticMetrics(NamedTuple):
    q: jax.Array
    h: jax.Array
    loss: jax.Array
    q_loss: jax.Array
    h_loss: jax.Array
    delta_l: jax.Array
    delta_r: jax.Array
    action_reg_loss: jax.Array
    h_reg_loss: jax.Array
    ensemble_grad_norms: jax.Array
    ensemble_weight_norms: jax.Array
    layer_grad_norms: jax.Array
    layer_weight_norms: jax.Array


@jax_u.jit
def get_ensemble_norm(tree: chex.ArrayTree):
    def _norm(x: jax.Array):
        return jnp.sqrt(jnp.sum(jnp.square(x)))

    def _tree_norm(x: chex.ArrayTree):
        leaves = jax.tree.leaves(x)
        norms = jax.tree.map(_norm, leaves)
        return sum(norms)

    return jax_u.vmap(_tree_norm)(tree)

def stable_rank(matrix: jax.Array):
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    sv_squared = singular_values**2
    return sv_squared.sum() / sv_squared.max()

def get_layer_names(params: chex.ArrayTree):
    keys = []
    def _inner(path: str, sub_params: chex.ArrayTree):
        if isinstance(sub_params, jax.Array):
            keys.append(path)

        elif isinstance(sub_params, dict):
            for key, value in sub_params.items():
                _inner(f"{path}/{key}", value)

    _inner('', params)
    return keys

@jax_u.jit
def get_stable_rank(params: chex.ArrayTree):
    leaves = jax.tree.leaves(params)
    ensemble = leaves[0].shape[0]
    names =  get_layer_names(params)

    # dim = 3 since the first dim is for the ensemble
    matrix_idxs =  [i for i, leaf in enumerate(leaves) if leaf.ndim == 3]
    matrix_leaves = [leaves[i] for i in matrix_idxs]
    matrix_names = [names[i] for i in matrix_idxs]

    def _sr(tree: chex.ArrayTree):
        return jax.tree.map(stable_rank, tree)

    stable_ranks = jax_u.vmap(_sr)(matrix_leaves)
    return [
        dict(
            zip(
                matrix_names,
                [sr[i] for sr in stable_ranks],
                strict=True,
            ),
        )
        for i in range(ensemble)
    ]


@partial(jax_u.jit, static_argnums=(1,))
def uniform_except(
    key: chex.PRNGKey,
    shape: tuple[int, ...],
    val: jax.Array,
    epsilon: jax.Array | float,
    minval: jax.Array | float,
    maxval: jax.Array | float,
):
    prop = jax.random.uniform(key, shape, minval=minval, maxval=maxval)

    def accept(prop: jax.Array):
        return (jnp.abs(prop - val) > epsilon) & (jnp.abs(prop - val) < 2 * epsilon)

    def keep_trying(carry: tuple[chex.PRNGKey, jax.Array, int]):
        _, prop, it = carry
        return jnp.logical_not(accept(prop)).any() & (it < 25)

    def body(carry: tuple[chex.PRNGKey, jax.Array, int]):
        key, prop, it = carry
        key, prop_key = jax.random.split(key)
        new = jax.random.uniform(prop_key, shape, minval=minval, maxval=maxval)
        x = jnp.where(
            accept(new),
            new,
            prop,
        )
        return key, x, it+1

    key, x, _ = jax.lax.while_loop(
        keep_trying,
        body,
        (key, prop, 0),
    )

    return x

@jax_u.jit
def get_layer_norms(params: chex.ArrayTree):
    leaves = jax.tree.leaves(params)
    ensemble = leaves[0].shape[0]
    names = get_layer_names(params)

    def _norm(x: jax.Array):
        return jnp.sqrt(jnp.sum(jnp.square(x)))

    def _tree_norm(x: chex.ArrayTree):
        return jax.tree.map(_norm, x)

    norms = jax_u.vmap(_tree_norm)(params)

    return [
        dict(
            zip(
                names,
                [n[i] for n in jax.tree.leaves(norms)],
                strict=True,
            ),
        )
        for i in range(ensemble)
    ]

def create_ensemble_dict(
    data: Any,
    metric_fn: Callable[[Any], list[dict[str, float]]],
    prefix: str = '',
) -> dict[str, dict[str, float]]:
    metrics = metric_fn(data)
    return {
        f"CRITIC{i}": {
            f"{prefix}{name}": float(value)
            for name, value in member_metrics.items()
        }
        for i, member_metrics in enumerate(metrics)
    }

def extract_metrics(
    metrics: dict[str, Any] | QRCCriticMetrics,
    metric_names: list[str] | None = None,
    array_processor: Callable[[jax.Array], float] | None = None,
    flatten_separator: str = "_",
) -> list[dict[str, float]]:
    if isinstance(metrics, QRCCriticMetrics):
        metrics = metrics._asdict()

    filtered = {k: v for k, v in metrics.items() if metric_names is None or k in metric_names}
    if not filtered:
        return []

    arrays = {k: v for k, v in filtered.items() if hasattr(v, 'shape') and hasattr(v, 'mean')}
    lists = {k: v for k, v in filtered.items() if not (hasattr(v, 'shape') and hasattr(v, 'mean'))}

    ensemble_data = jax_u.extract_axis(arrays, axis=0) if arrays else []
    ensemble_size = len(lists.get(next(iter(lists)), [])) if lists else 0
    if not ensemble_data and ensemble_size > 0:
        ensemble_data = [{} for _ in range(ensemble_size)]

    processor = array_processor or (lambda x: float(x.mean().squeeze()))

    result = []
    for i, member in enumerate(ensemble_data):
        member_dict = {k: processor(v) for k, v in member.items()}

        for name, lst in lists.items():
            if i < len(lst):
                val = lst[i]
                if isinstance(val, dict):
                    flattened = dict_u.flatten_nested_dict(val, separator=flatten_separator)
                    member_dict.update({f"{name}_{k}": float(v) for k, v in flattened.items()})
                else:
                    member_dict[name] = float(val)

        result.append(member_dict)

    return result
