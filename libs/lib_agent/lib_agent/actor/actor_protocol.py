from typing import NamedTuple, Protocol

import chex
import jax


class PolicyState(NamedTuple):
    params: chex.ArrayTree
    opt_state: chex.ArrayTree | None = None
    group_opt_states: dict[str, chex.ArrayTree] | None = None


class ActorState(Protocol):
    actor: PolicyState


class PolicyOutputs(NamedTuple):
    mu: jax.Array
    sigma: jax.Array


class ValueEstimator(Protocol):
    def __call__(
        self,
        params: chex.ArrayTree,
        rng: chex.PRNGKey,
        x: jax.Array,
        a: jax.Array,
    ) -> jax.Array: ...
