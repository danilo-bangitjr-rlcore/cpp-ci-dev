from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lib_agent.buffer.datatypes import DataMode
from lib_agent.buffer.mixed_history_buffer import MaskedABDistribution, MixedHistoryBuffer
from lib_agent.buffer.recency_bias_buffer import MaskedUGDistribution


class FakeTransition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray


def create_test_transition(i: int):
    return FakeTransition(
        state=jnp.array([i]),
        action=jnp.array([i]),
        reward=jnp.array([float(i)]),
    )


def test_feed_online_mode():
    for _ in range(100):
        buffer = MixedHistoryBuffer[FakeTransition](
            ensemble=1,
            max_size=100,
            batch_size=10,
            n_most_recent=2,
            seed=42,
        )
        online_transitions = [create_test_transition(i) for i in range(5)]
        idxs = buffer.feed(online_transitions, DataMode.ONLINE)
        assert len(idxs) == 5

        offline_transitions = [create_test_transition(i) for i in range(5, 10)]
        idxs = buffer.feed(offline_transitions, DataMode.OFFLINE)

        samples = buffer.sample()
        assert samples.state.shape == (1, 10, 1)


def test_masked_ab_distribution():
    dist = MaskedABDistribution(support=100, left_prob=0.5, mask_prob=0.5)
    assert dist.size() == 0

    elements = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(dist.probs(elements), np.zeros(5))

    ensemble_mask = np.array([1, 1, 0, 0, 1], dtype=bool)
    dist.update(np.random.default_rng(42), elements, DataMode.ONLINE, ensemble_mask)
    assert dist.size() == 3

    # check that probabilities are non-zero for masked elements and zero for others
    probs = dist.probs(elements)
    assert np.all(probs[ensemble_mask] > 0)
    assert np.all(probs[~ensemble_mask] == 0)
    assert probs.sum() > 0

    new_elements = np.array([5, 6])
    new_ensemble_mask = np.array([1, 0], dtype=bool)

    dist.update(np.random.default_rng(42), new_elements, DataMode.OFFLINE, new_ensemble_mask)

    all_elements = np.concatenate((elements, new_elements))
    all_probs = dist.probs(all_elements)
    assert np.allclose(all_probs.sum(), 1.0)
    assert dist.size() == 4

def test_masked_ug_distribution():
    dist = MaskedUGDistribution(support=100, left_prob=0.5, mask_prob=0.5)
    assert dist.size() == 0

    elements = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(dist.probs(elements), np.zeros(5))

    ensemble_mask = np.array([1, 1, 0, 0, 1], dtype=bool)
    dist.update_uniform(elements, ensemble_mask)
    dist.update_geometric(elements, ensemble_mask)
    assert dist.size() == 3
    assert np.allclose(dist.probs(elements), ensemble_mask/3)

    dist.discount_geometric(0.5)
    assert np.allclose(dist.probs(elements), ensemble_mask/3)

    new_elements = np.array([5, 6])
    new_ensemble_mask = np.array([1, 0], dtype=bool)

    dist.update_uniform(new_elements, new_ensemble_mask)
    dist.update_geometric(new_elements, new_ensemble_mask)

    assert np.allclose(
        dist.probs(np.concatenate((elements, new_elements))),
        np.array([0.225, 0.225, 0., 0., 0.225, 0.325, 0.]),
    )

    assert dist.size() == 4

def test_mixed_history_buffer_online_offline_mixing():
    buffer = MixedHistoryBuffer(
        ensemble=1,
        max_size=100,
        batch_size=10,
        n_most_recent=1,
        seed=42,
        online_weight=0.75,
    )

    online_transitions = [create_test_transition(i) for i in range(5)]
    buffer.feed(online_transitions, DataMode.ONLINE)

    offline_transitions = [create_test_transition(i) for i in range(5, 10)]
    buffer.feed(offline_transitions, DataMode.OFFLINE)

    assert buffer.size == 10
    assert buffer.ensemble_sizes[0] > 0


def test_mixed_history_buffer_ensemble():
    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=2,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
        ensemble_probability=0.5,
    )

    transitions = [create_test_transition(i) for i in range(10)]
    buffer.feed(transitions, DataMode.ONLINE)

    assert buffer.size == 10
    assert len(buffer.ensemble_sizes) == 2
    assert all(size > 0 for size in buffer.ensemble_sizes)

    samples = buffer.sample()
    assert samples.state.shape == (2, 5, 1)


def test_mixed_history_buffer_sampleable():
    buffer = MixedHistoryBuffer(
        ensemble=1,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
    )

    assert not buffer.is_sampleable

    transitions = [create_test_transition(0)]
    buffer.feed(transitions, DataMode.ONLINE)

    assert buffer.is_sampleable


def test_mixed_history_buffer_get_batch():
    buffer = MixedHistoryBuffer[FakeTransition](
        ensemble=1,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
    )

    transitions = [create_test_transition(i) for i in range(3)]
    idxs = buffer.feed(transitions, DataMode.ONLINE)

    batch = buffer.get_batch(idxs)
    assert batch.state.shape == (3, 1)
    assert batch.action.shape == (3, 1)
    assert batch.reward.shape == (3, 1)
