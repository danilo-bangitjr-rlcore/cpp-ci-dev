"""
Benchmark tests for MixedHistoryBuffer operations.
"""

from typing import NamedTuple

import jax.numpy as jnp
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.buffer.datatypes import DataMode
from lib_agent.buffer.mixed_history_buffer import MixedHistoryBuffer


class FakeTransition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: float


def create_test_transition(i: int) -> FakeTransition:
    return FakeTransition(
        state=jnp.array([i], dtype=jnp.float32),
        action=jnp.array([i % 10]),
        reward=float(i),
    )


def test_mixed_history_buffer_feed(benchmark: BenchmarkFixture):
    """Benchmark mixed feeding online and offline data to buffer."""

    # Setup outside of benchmark - this is expensive and should not be measured
    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=4,
        max_size=100000,
        batch_size=256,
        online_weight=0.75,
        seed=42,
    )

    # Create test data outside of benchmark
    online_transitions = [create_test_transition(i) for i in range(1000)]
    offline_transitions = [create_test_transition(i) for i in range(1000, 2000)]

    def feed_mixed_data():
        # Only measure the actual feed operations
        buffer.feed(online_transitions, DataMode.ONLINE)
        buffer.feed(offline_transitions, DataMode.OFFLINE)
        return buffer

    buffer = benchmark(feed_mixed_data)
    assert sum(buffer.ensemble_sizes) > 0


def test_mixed_history_buffer_sampling_small(benchmark: BenchmarkFixture):
    """Benchmark sampling from small mixed history buffer."""

    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=2,
        max_size=10000,
        batch_size=128,
        online_weight=0.6,
        seed=42,
    )

    # Populate with mixed data
    online_transitions = [create_test_transition(i) for i in range(1000)]
    offline_transitions = [create_test_transition(i) for i in range(1000, 1500)]

    buffer.feed(online_transitions, DataMode.ONLINE)
    buffer.feed(offline_transitions, DataMode.OFFLINE)

    def sample_data():
        return buffer.sample()

    batch = benchmark(sample_data)
    assert batch.state.shape == (2, 128, 1)
    assert batch.action.shape == (2, 128, 1)
    assert batch.reward.shape == (2, 128)


def test_mixed_history_buffer_sampling_large(benchmark: BenchmarkFixture):
    """Benchmark sampling from large mixed history buffer."""

    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=8,
        max_size=50000,
        batch_size=512,
        online_weight=0.8,
        seed=42,
    )

    # Populate with large mixed dataset
    online_transitions = [create_test_transition(i) for i in range(5000)]
    offline_transitions = [create_test_transition(i) for i in range(5000, 8000)]

    buffer.feed(online_transitions, DataMode.ONLINE)
    buffer.feed(offline_transitions, DataMode.OFFLINE)

    def sample_large():
        return buffer.sample()

    batch = benchmark(sample_large)
    assert batch.state.shape == (8, 512, 1)
    assert batch.action.shape == (8, 512, 1)
    assert batch.reward.shape == (8, 512)


def test_mixed_history_buffer_get_batch(benchmark: BenchmarkFixture):
    """Benchmark get_batch with specific indices."""

    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=6,
        max_size=30000,
        batch_size=512,
        online_weight=0.7,
        seed=42,
    )

    # Populate with data
    for i in range(3000):
        fake_transition = create_test_transition(i)
        data_mode = DataMode.ONLINE if i % 2 == 0 else DataMode.OFFLINE
        buffer.feed([fake_transition], data_mode)

    # Pre-generate indices
    import numpy as np
    indices = np.random.randint(0, buffer.size, size=(6, 256))

    def get_specific_batch():
        return buffer.get_batch(indices)

    batch = benchmark(get_specific_batch)
    assert batch.state.shape == (6, 256, 1)
    assert batch.action.shape == (6, 256, 1)
    assert batch.reward.shape == (6, 256)
