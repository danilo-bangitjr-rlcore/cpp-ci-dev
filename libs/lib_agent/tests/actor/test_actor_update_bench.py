from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lib_utils.named_array import NamedArray
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.actor.percentile_actor import PAConfig, PercentileActor
from lib_agent.buffer.buffer import State


class FakeValueEstimator:
    """Simple value estimator for benchmarking."""
    def __call__(self, params: Any, rng: jax.Array, x: jax.Array, a: jax.Array) -> jax.Array:
        # Return scalar value for each action sample
        # If a has multiple dimensions beyond the action_dim, collapse them
        if a.ndim > 1:
            # Flatten action dimensions to return one value per sample
            return jnp.sum(a, axis=-1)
        return jnp.sum(a)


def _create_test_actor(state_dim: int, action_dim: int, num_samples: int = 256):
    """Create a percentile actor for testing."""
    config = PAConfig(
        name="PA",
        num_samples=num_samples,
        actor_percentile=0.8,
        proposal_percentile=0.5,
        uniform_weight=0.1,
        actor_lr=1e-3,
        proposal_lr=1e-3,
        mu_multiplier=1.0,
        sigma_multiplier=1.0,
        max_action_stddev=1.0,
        sort_noise=0.0,
    )

    return PercentileActor(
        cfg=config,
        seed=42,
        state_dim=state_dim,
        action_dim=action_dim,
    )


class FakeActorBatch(NamedTuple):
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


class BenchmarkConfig(NamedTuple):
    """Configuration for actor update benchmark tests."""
    state_dim: int
    action_dim: int
    batch_size: int
    num_samples: int
    iterations: int


@pytest.mark.parametrize(
    "config",
    [
        BenchmarkConfig(state_dim=64, action_dim=4, batch_size=128, num_samples=256, iterations=5),  # small
        BenchmarkConfig(state_dim=128, action_dim=8, batch_size=512, num_samples=512, iterations=3), # large
    ],
    ids=["small", "large"],
)
def test_actor_policy_update(benchmark: BenchmarkFixture, config: BenchmarkConfig) -> None:
    """Benchmark actor policy update with parameterized configurations."""
    rng = np.random.default_rng(0)

    actor = _create_test_actor(config.state_dim, config.action_dim, num_samples=config.num_samples)

    # Initialize actor
    init_rng = jax.random.PRNGKey(42)
    single_state = jnp.array(rng.random((config.state_dim,)))
    actor_state = actor.init_state(init_rng, single_state)

    # Create value estimator and its parameters
    value_estimator = FakeValueEstimator()
    value_estimator_params = {"dummy": jnp.array(1.0)}

    # Create actor batch data with ensemble dimension
    ensemble_size = 1
    states = jnp.array(rng.random((ensemble_size, config.batch_size, config.state_dim)))
    actions = jnp.array(rng.random((config.batch_size, config.action_dim)))
    rewards = jnp.array(rng.random((config.batch_size,)))
    next_states = jnp.array(rng.random((ensemble_size, config.batch_size, config.state_dim)))
    gammas = jnp.full((config.batch_size,), 0.99)

    # Create state objects with ensemble dimension
    batch_state = State(
        features=NamedArray.unnamed(states),
        a_lo=jnp.tile(jnp.array([-1.0] * config.action_dim), (ensemble_size, config.batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * config.action_dim), (ensemble_size, config.batch_size, 1)),
        dp=jnp.ones((ensemble_size, config.batch_size, 1)),
        last_a=jnp.tile(actions, (ensemble_size, 1, 1)),
    )

    next_batch_state = State(
        features=NamedArray.unnamed(next_states),
        a_lo=jnp.tile(jnp.array([-1.0] * config.action_dim), (ensemble_size, config.batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * config.action_dim), (ensemble_size, config.batch_size, 1)),
        dp=jnp.ones((ensemble_size, config.batch_size, 1)),
        last_a=jnp.tile(actions, (ensemble_size, 1, 1)),
    )

    actor_batch = FakeActorBatch(
        state=batch_state,
        action=actions,
        reward=rewards,
        next_state=next_batch_state,
        gamma=gammas,
    )

    def _inner(
        actor: PercentileActor,
        state: Any,
        ve: FakeValueEstimator,
        ve_params: Any,
        batch: FakeActorBatch,
    ):
        for _ in range(config.iterations):
            _new_state, metrics = actor.update(state, ve, ve_params, batch)
            # Force computation
            _ = metrics.actor_loss.sum()

    benchmark(_inner, actor, actor_state, value_estimator, value_estimator_params, actor_batch)
