from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
from lib_utils import dict as dict_u

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
class RollingResetConfig:
    reset_period: int = 10000
    warm_up_steps: int = 1000


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
    rolling_reset_config: RollingResetConfig = field(default_factory=RollingResetConfig)


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
