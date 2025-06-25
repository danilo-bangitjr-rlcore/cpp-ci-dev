from datetime import UTC, datetime
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig


class FakeTransition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: float
    next_state: jnp.ndarray
    done: bool

def test_recency_bias_buffer_basic():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1000,  # 1ms
            gamma=[0.99],
            effective_episodes=[100],
            ensemble=1,
            uniform_weight=0.5,
            ensemble_probability=0.5,
            max_size=1000,
        ),
    )
    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
        np.datetime64('2024-01-01T00:00:02'),
    ])

    for i in range(3):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, timestamps[i])

    assert buffer.size == 3

    assert buffer.ensemble_masks.shape == (1, 1000)  # (n_ensemble, max_size)
    assert np.sum(buffer.ensemble_masks[:, 3:]) == 0
    assert np.all(np.sum(buffer.ensemble_masks[:, :3], axis=0) >= 1)


def test_recency_bias_buffer_weights():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1000,
            gamma=[0.99],
            effective_episodes=[100],
            ensemble=1,
            uniform_weight=0.5,
            ensemble_probability=1.0,
            max_size=1000,
        ),
    )

    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
        np.datetime64('2024-01-01T00:00:02'),
    ])

    for i in range(3):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, timestamps[i])

    probs = buffer.get_probability(0, np.array([0, 1, 2]))

    assert probs[2] > probs[1] > probs[0]


def test_recency_bias_buffer_discount():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1000,  # 1ms
            gamma=[0.99],
            effective_episodes=[100],
            ensemble=1,
            uniform_weight=0.5,
            ensemble_probability=1.0,
            max_size=1000,
        ),
    )

    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
    ])

    for i in range(2):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, timestamps[i])
    initial_probs = buffer.get_probability(0, np.array([0, 1]))

    later_timestamp = np.datetime64('2024-01-01T00:00:10')
    transition = FakeTransition(
        state=jnp.array([2]),
        action=jnp.array([2]),
        reward=2.0,
        next_state=jnp.array([3]),
        done=False,
    )
    buffer.add(transition, later_timestamp)
    new_probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert new_probs[0] < initial_probs[0]
    assert new_probs[1] < initial_probs[1]
    assert new_probs[2] > new_probs[1] > new_probs[0]


def test_recency_bias_buffer_datetime_timestamps():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1000,  # 1ms
            gamma=[0.99],
            effective_episodes=[100],
            ensemble=1,
            uniform_weight=0.5,
            ensemble_probability=1.0,
            max_size=1000,
        ),
    )

    timestamps = [
        datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC),
    ]

    for i, ts in enumerate(timestamps):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, ts)

    probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert np.all(np.diff(probs) > 0)
    assert probs[2] / probs[0] > 1.5


def test_recency_bias_buffer_integer_timestamps():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1000,  # 1ms
            gamma=[0.99],
            effective_episodes=[100],
            ensemble=1,
            uniform_weight=0.5,
            ensemble_probability=1.0,
            max_size=1000,
        ),
    )

    timestamps = [0, 1, 2]

    for i, ts in enumerate(timestamps):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, ts)

    probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert probs[2] > probs[1] > probs[0]
    assert probs[2] / probs[0] < 1.5


def test_recency_bias_buffer_different_discounts():
    buffer = RecencyBiasBuffer(
        RecencyBiasBufferConfig(
            obs_period=1,
            gamma=[0.9, 0.99],
            effective_episodes=[5, 10],
            ensemble=2,
            uniform_weight=0.1,
            ensemble_probability=1.0,
            max_size=1000,
            batch_size=1,
        ),
    )

    n_transitions = 10
    for i in range(n_transitions):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition, i)

    probs_0 = buffer.get_probability(0, np.arange(n_transitions))
    probs_1 = buffer.get_probability(1, np.arange(n_transitions))

    probs_0 = probs_0 / probs_0.sum()
    probs_1 = probs_1 / probs_1.sum()

    ratio_0 = probs_0[-1] / probs_0[0]
    ratio_1 = probs_1[-1] / probs_1[0]
    assert ratio_0 > ratio_1, "First ensemble should have stronger recency bias"
    assert ratio_0 > 1.0 and ratio_1 > 1.0, "Both ensembles should show recency bias"

    assert np.all(np.diff(probs_0) > 0), "First ensemble probabilities should increase with recency"
    assert np.all(np.diff(probs_1) > 0), "Second ensemble probabilities should increase with recency"

