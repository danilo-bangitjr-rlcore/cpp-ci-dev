from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.buffer.buffer import EnsembleReplayBuffer


class FakeTransition(NamedTuple):
    """Simple transition for benchmarking."""
    state: jax.Array
    action: jax.Array
    reward: float
    gamma: float
    next_state: jax.Array

    @property
    def a_lo(self):
        return jnp.array([-1.0, -1.0])

    @property
    def a_hi(self):
        return jnp.array([1.0, 1.0])

    @property
    def next_a_lo(self):
        return jnp.array([-1.0, -1.0])

    @property
    def next_a_hi(self):
        return jnp.array([1.0, 1.0])

    @property
    def dp(self):
        return False

    @property
    def next_dp(self):
        return False

    @property
    def last_a(self):
        return self.action

    @property
    def state_dim(self):
        return self.state.shape[0]

    @property
    def action_dim(self):
        return self.action.shape[0]


class BufferBenchmarkConfig(NamedTuple):
    """Configuration for buffer benchmark tests."""
    ensemble: int
    max_size: int
    batch_size: int
    state_dim: int
    action_dim: int
    n_most_recent: int
    prefill_count: int
    iterations: int


@pytest.mark.parametrize(
    "config",
    [
        # small
        BufferBenchmarkConfig(
            ensemble=2, max_size=10000, batch_size=128, state_dim=64, action_dim=2,
            n_most_recent=5, prefill_count=5000, iterations=20,
        ),
        # medium
        BufferBenchmarkConfig(
            ensemble=4, max_size=50000, batch_size=256, state_dim=128, action_dim=2,
            n_most_recent=10, prefill_count=25000, iterations=15,
        ),
        # large
        BufferBenchmarkConfig(
            ensemble=8, max_size=100000, batch_size=512, state_dim=256, action_dim=2,
            n_most_recent=20, prefill_count=50000, iterations=10,
        ),
    ],
    ids=["small", "medium", "large"],
)
def test_ensemble_buffer_sampling(benchmark: BenchmarkFixture, config: BufferBenchmarkConfig) -> None:
    """
    Benchmark ensemble buffer sampling with parameterized configurations.
    """
    rng = np.random.default_rng(0)
    buffer = EnsembleReplayBuffer[FakeTransition](
        ensemble=config.ensemble,
        max_size=config.max_size,
        ensemble_probability=0.5,
        batch_size=config.batch_size,
        seed=42,
        n_most_recent=config.n_most_recent,
    )

    fake_transition = FakeTransition(
        state=jnp.array(rng.random((config.state_dim,))),
        action=jnp.array(rng.random((config.action_dim,))),
        reward=0.1,
        gamma=0.99,
        next_state=jnp.array(rng.random((config.state_dim,))),
    )

    # Prefill buffer to ensure meaningful sampling
    for _ in range(config.prefill_count):
        buffer.add(fake_transition)

    def _inner(buffer: EnsembleReplayBuffer[FakeTransition]):
        for _ in range(config.iterations):
            batch = buffer.sample()
            # Access the data to ensure computation happens
            _ = batch.state.sum()

    benchmark(_inner, buffer)


def test_ensemble_buffer_add_performance(benchmark: BenchmarkFixture):
    """Benchmark the add operation performance for ensemble buffers."""
    rng = np.random.default_rng(0)
    buffer = EnsembleReplayBuffer[FakeTransition](
        ensemble=4,
        max_size=100000,
        ensemble_probability=0.5,
        batch_size=256,
        seed=42,
    )

    fake_transition = FakeTransition(
        state=jnp.array(rng.random((128,))),
        action=jnp.array(rng.random((2,))),
        reward=0.1,
        gamma=0.99,
        next_state=jnp.array(rng.random((128,))),
    )

    def _inner(buffer: EnsembleReplayBuffer, transition: FakeTransition):
        for _ in range(1000):
            buffer.add(transition)

    benchmark(_inner, buffer, fake_transition)
