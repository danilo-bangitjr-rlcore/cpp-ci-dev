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
    gamma: float
    a_lo: jnp.ndarray
    a_hi: jnp.ndarray
    next_a_lo: jnp.ndarray
    next_a_hi: jnp.ndarray
    dp: bool
    next_dp: bool
    last_a: jnp.ndarray
    state_dim: int
    action_dim: int


def create_test_transition(i: int) -> FakeTransition:
    return FakeTransition(
        state=jnp.array([i]),
        action=jnp.array([i]),
        reward=float(i),
        next_state=jnp.array([i+1]),
        gamma=0.99,
        a_lo=jnp.array([i-0.5]),
        a_hi=jnp.array([i+0.5]),
        next_a_lo=jnp.array([i+0.5]),
        next_a_hi=jnp.array([i+1.5]),
        dp=True,
        next_dp=True,
        last_a=jnp.array([i-1]),
        state_dim=1,
        action_dim=1,
    )


def test_recency_bias_buffer_basic():
    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99],
        effective_episodes=[100],
        n_ensemble=1,
        uniform_weight=0.5,
        ensemble_probability=0.5,
        max_size=1000,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
    )
    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
        np.datetime64('2024-01-01T00:00:02'),
    ])

    for i in range(3):
        transition = create_test_transition(i)
        buffer.add(transition, timestamps[i])

    assert buffer.size == 3

    assert buffer.ensemble_masks.shape == (1, 1000)
    assert np.sum(buffer.ensemble_masks[:, 3:]) == 0
    assert np.all(np.sum(buffer.ensemble_masks[:, :3], axis=0) >= 1)


def test_recency_bias_buffer_weights():
    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99],
        effective_episodes=[100],
        n_ensemble=1,
        uniform_weight=0.5,
        ensemble_probability=1.0,
        max_size=1000,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
    )

    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
        np.datetime64('2024-01-01T00:00:02'),
    ])

    for i in range(3):
        transition = create_test_transition(i)
        buffer.add(transition, timestamps[i])

    probs = buffer.get_probability(0, np.array([0, 1, 2]))

    assert probs[2] > probs[1] > probs[0]


def test_recency_bias_buffer_discount():
    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99],
        effective_episodes=[100],
        n_ensemble=1,
        uniform_weight=0.5,
        ensemble_probability=1.0,
        max_size=1000,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
    )

    timestamps = np.array([
        np.datetime64('2024-01-01T00:00:00'),
        np.datetime64('2024-01-01T00:00:01'),
    ])

    for i in range(2):
        transition = create_test_transition(i)
        buffer.add(transition, timestamps[i])
    initial_probs = buffer.get_probability(0, np.array([0, 1]))

    later_timestamp = np.datetime64('2024-01-01T00:00:10')
    transition = create_test_transition(2)
    buffer.add(transition, later_timestamp)
    new_probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert new_probs[0] < initial_probs[0]
    assert new_probs[1] < initial_probs[1]
    assert new_probs[2] > new_probs[1] > new_probs[0]


def test_recency_bias_buffer_datetime_timestamps():
    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99],
        effective_episodes=[100],
        n_ensemble=1,
        uniform_weight=0.5,
        ensemble_probability=1.0,
        max_size=1000,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
    )

    timestamps = [
        datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC),
    ]

    for i, ts in enumerate(timestamps):
        transition = create_test_transition(i)
        buffer.add(transition, ts)

    probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert np.all(np.diff(probs) > 0)
    assert probs[2] / probs[0] > 1.5


def test_recency_bias_buffer_integer_timestamps():
    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99],
        effective_episodes=[100],
        n_ensemble=1,
        uniform_weight=0.5,
        ensemble_probability=1.0,
        max_size=1000,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
    )

    timestamps = [0, 1, 2]

    for i, ts in enumerate(timestamps):
        transition = create_test_transition(i)
        buffer.add(transition, ts)

    probs = buffer.get_probability(0, np.array([0, 1, 2]))
    assert probs[2] > probs[1] > probs[0]
    assert probs[2] / probs[0] < 1.5


def test_recency_bias_buffer_different_discounts():
    cfg = RecencyBiasBufferConfig(
        obs_period=1,
        gamma=[0.9, 0.99],
        effective_episodes=[5, 10],
        n_ensemble=2,
        uniform_weight=0.1,
        ensemble_probability=1.0,
        max_size=1000,
        batch_size=1,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        n_ensemble=cfg.n_ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
        batch_size=cfg.batch_size,
    )

    n_transitions = 10
    for i in range(n_transitions):
        transition = create_test_transition(i)
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

