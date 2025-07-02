from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lib_agent.buffer.mixed_history_buffer import MaskedABDistribution, MixedHistoryBuffer
from lib_agent.buffer.recency_bias_buffer import MaskedUGDistribution


class FakeTransition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: float
    next_state: jnp.ndarray
    done: bool


class FakeJaxTransition:
    def __init__(self, state, action, reward, next_state, prior_action, steps, n_step_reward, n_step_gamma, state_dim, action_dim):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.prior = FakePrior(prior_action)
        self.steps = steps
        self.n_step_reward = n_step_reward
        self.n_step_gamma = n_step_gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

    @property
    def gamma(self):
        return self.n_step_gamma


class FakePrior:
    def __init__(self, action):
        self.action = action


class FakeStep:
    def __init__(self, action_lo, action_hi, dp):
        self.action_lo = action_lo
        self.action_hi = action_hi
        self.dp = dp


def make_test_step(start: int) -> FakeStep:
    return FakeStep(
        action_lo=jnp.array([start]),
        action_hi=jnp.array([start + 1]),
        dp=True,
    )


def make_test_transition(start: int, length: int) -> FakeJaxTransition:
    steps = [make_test_step(start + i) for i in range(length + 1)]
    return FakeJaxTransition(
        state=jnp.array([start]),
        action=jnp.array([start]),
        reward=1.0,
        next_state=jnp.array([start + 1]),
        prior_action=jnp.array([start - 1]),
        steps=steps,
        n_step_reward=1.0,
        n_step_gamma=0.99,
        state_dim=1,
        action_dim=1,
    )


def make_test_transitions(start: int, num: int, length: int) -> list[FakeJaxTransition]:
    return [
        make_test_transition(i, length)
        for i in range(start, start + num)
    ]


def test_feed_online_mode():
    for _ in range(100):
        buffer = MixedHistoryBuffer(
            n_ensemble=1,
            max_size=100,
            batch_size=10,
            n_most_recent=2,
            seed=42,
        )
        online_transitions = make_test_transitions(0, 5, 1)
        idxs = buffer.feed(online_transitions, "online")
        assert len(idxs) == 5

        offline_transitions = make_test_transitions(5, 5, 1)
        idxs = buffer.feed(offline_transitions, "offline")

        samples = buffer.sample()
        assert samples.state.shape == (1, 10, 1)


def test_masked_ab_distribution():
    dist = MaskedABDistribution(support=100, left_prob=0.5, mask_prob=0.5)
    assert dist.size() == 0

    elements = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(dist.probs(elements), np.zeros(5))

    ensemble_mask = np.array([1, 1, 0, 0, 1], dtype=bool)
    dist.update(np.random.default_rng(42), elements, "online", ensemble_mask)
    assert dist.size() == 3

    # check that probabilities are non-zero for masked elements and zero for others
    probs = dist.probs(elements)
    assert np.all(probs[ensemble_mask] > 0)
    assert np.all(probs[~ensemble_mask] == 0)
    assert probs.sum() > 0

    new_elements = np.array([5, 6])
    new_ensemble_mask = np.array([1, 0], dtype=bool)

    dist.update(np.random.default_rng(42), new_elements, "offline", new_ensemble_mask)

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
        n_ensemble=1,
        max_size=100,
        batch_size=10,
        n_most_recent=1,
        seed=42,
        online_weight=0.75,
    )

    online_transitions = make_test_transitions(0, 5, 1)
    buffer.feed(online_transitions, "online")

    offline_transitions = make_test_transitions(5, 5, 1)
    buffer.feed(offline_transitions, "offline")

    assert buffer.size == 10
    assert buffer.ensemble_sizes[0] > 0


def test_mixed_history_buffer_ensemble():
    buffer = MixedHistoryBuffer(
        n_ensemble=2,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
        ensemble_probability=0.5,
    )

    transitions = make_test_transitions(0, 10, 1)
    buffer.feed(transitions, "online")

    assert buffer.size == 10
    assert len(buffer.ensemble_sizes) == 2
    assert all(size > 0 for size in buffer.ensemble_sizes)

    samples = buffer.sample()
    assert samples.state.shape == (2, 5, 1)


def test_mixed_history_buffer_sampleable():
    buffer = MixedHistoryBuffer(
        n_ensemble=1,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
    )

    assert not buffer.is_sampleable

    transitions = make_test_transitions(0, 1, 1)
    buffer.feed(transitions, "online")

    assert buffer.is_sampleable


def test_mixed_history_buffer_get_batch():
    buffer = MixedHistoryBuffer(
        n_ensemble=1,
        max_size=100,
        batch_size=5,
        n_most_recent=1,
        seed=42,
    )

    transitions = make_test_transitions(0, 3, 1)
    idxs = buffer.feed(transitions, "online")

    batch = buffer.get_batch(idxs)
    assert batch.state.shape == (3, 1)
    assert batch.action.shape == (3, 1)
    assert batch.reward.shape == (3,)
