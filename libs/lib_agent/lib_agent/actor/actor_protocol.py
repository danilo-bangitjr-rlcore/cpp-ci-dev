from dataclasses import dataclass
from typing import NamedTuple, Protocol

import chex
import distrax
import jax
import jax.numpy as jnp

from lib_agent.buffer.datatypes import State, Transition


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


class Actor(Protocol):
    """Protocol defining the interface for actor implementations."""

    def __init__(self, cfg: ActorConfig, seed: int, state_dim: int, action_dim: int):
        ...

    def init_state(self, rng: chex.PRNGKey, x: jax.Array) -> ActorState:
        """Initialize the actor parameters."""
        ...

    def get_actions(
        self,
        actor_params: chex.ArrayTree,
        state: State,
        n: int = 1,
        std_devs: float = jnp.inf,
    ) -> tuple[jax.Array, dict[str, float]]:
        """Sample n actions from the actor distribution at the given state"""
        ...

    def get_dist(self, actor_params: chex.ArrayTree, state: State) -> distrax.Distribution:
        """Get actor distribution."""
        ...

    def get_probs(self, params: chex.ArrayTree, state: State, actions: jax.Array) -> jax.Array:
        """Get actor probabilities for the given state and actions."""
        ...

    def get_log_probs(self, params: chex.ArrayTree, state: State, actions: jax.Array) -> jax.Array:
        """Get actor log probabilities for the given state and actions."""
        ...

    def update(
        self,
        update_rng: chex.PRNGKey,
        dist_state: ActorState,
        value_estimator: ValueEstimator,
        value_estimator_params: chex.ArrayTree,
        transitions: Transition,
    ) -> tuple[ActorState, ActorUpdateMetrics]:
        """Update the policy distributions at the states in the given state transitions."""
        ...

    def initialize_to_nominal_action(
        self,
        rng: chex.PRNGKey,
        policy_state: PolicyState,
        nominal_actions: jax.Array,
        state_dim: int,
    ) -> PolicyState:
        """Initialize the actor to be maximized at a nominal action."""
        ...
