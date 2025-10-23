from dataclasses import dataclass
from typing import NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp


class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree | None = None
    group_opt_states: dict[str, chex.ArrayTree] | None = None


class ActorState(Protocol):
    actor: PolicyState


class PolicyOutputs(NamedTuple):
    mu: jax.Array
    sigma: jax.Array


class ActorUpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    actor_grad_norm: jax.Array


class ValueEstimator(Protocol):
    def __call__(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        x: jax.Array,
        a: jax.Array,
    ) -> jax.Array: ...


@dataclass
class ActorConfig:
    name: str
    actor_lr: float = 0.0001
    mu_multiplier: float = 1.0
    sigma_multiplier: float = 1.0
    max_action_stddev: float = jnp.inf
