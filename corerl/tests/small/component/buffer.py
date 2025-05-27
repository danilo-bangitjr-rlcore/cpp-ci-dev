from datetime import datetime, timedelta

import numpy as np
import torch

from corerl.component.buffer import (
    MaskedUGDistribution,
    MixedHistoryBuffer,
    MixedHistoryBufferConfig,
    RecencyBiasBuffer,
    RecencyBiasBufferConfig,
)
from corerl.data_pipeline.datatypes import DataMode, Transition
from corerl.state import AppState
from tests.small.data_pipeline.test_transition_filter import make_test_step


def make_test_transition(start: int, len: int) -> Transition:
    steps = [make_test_step(start + i) for i in range(len+1)]
    transition = Transition(steps, 1, .99)
    return transition

def make_test_transitions(start:int, num: int, len: int) -> list[Transition]:
    transitions = []
    for i in range(start, start+num):
        transitions.append(make_test_transition(i, len))
    return transitions

def test_feed_online_mode(dummy_app_state: AppState):
    buffer_cfg = MixedHistoryBufferConfig(
        seed=42,
        memory=100,
        batch_size=10,
        n_most_recent=2,
    )

    for _ in range(100):
        buffer = MixedHistoryBuffer(buffer_cfg, dummy_app_state)
        online_transitions = make_test_transitions(0, 5, 1)
        idxs = buffer.feed(online_transitions, DataMode.ONLINE)
        assert len(idxs) == 5

        # now feed some transitions with a different mode
        offline_transitions = make_test_transitions(5, 5, 1)
        idxs = buffer.feed(offline_transitions, DataMode.OFFLINE)

        samples = buffer.sample()

        for batch in samples:
            assert (batch.prior.state[:2, 0] == torch.Tensor([4, 3])).all()

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
        np.array([0.225, 0.225, 0., 0., 0.225, 0.325, 0.])
    )

    assert dist.size() == 4

def add_ts(transitions: list[Transition], times: list[datetime]):
    for i, t in enumerate(transitions):
        for s in t.steps:
            s.timestamp = times[i]
    return transitions

def test_recency_bias_buffer_discounting(dummy_app_state: AppState):
    obs_period = dummy_app_state.cfg.interaction.obs_period
    buffer_cfg = RecencyBiasBufferConfig(
        seed=42,
        memory=100,
        batch_size=10,
        n_most_recent=1,
        obs_period=obs_period,
        gamma=0.9,
        ensemble=1,
        ensemble_probability=1,
        uniform_weight=0.0,
        effective_episodes=2,
    )

    buffer = RecencyBiasBuffer(buffer_cfg, dummy_app_state)
    expected_discount = np.power(0.9, 1/2)
    assert buffer._discount_factor == expected_discount
    now = datetime(2000, 1, 1)
    time_increase = timedelta(seconds=1)

    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now])
    idxs = buffer.feed(transitions, DataMode.ONLINE)
    assert len(idxs) == 1
    expected_probs = np.ones(1)
    assert np.allclose(buffer.get_probability(0, idxs), expected_probs)

    now += time_increase

    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now])
    idxs = buffer.feed(transitions, DataMode.ONLINE)
    assert len(idxs) == 1
    expected_weights = np.array([expected_discount, 1])
    expected_probs =  expected_weights / expected_weights.sum()
    assert np.allclose(
        buffer.get_probability(0, np.array([0, 1])), # probability of last two elements
        expected_probs,
    )

    # add an element which is older then the previous transition.
    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now-time_increase])
    idxs = buffer.feed(transitions, DataMode.ONLINE)
    assert len(idxs) == 1
    # should not discount previous transition but should discount previously added transition
    expected_weights = np.array([expected_discount, 1, expected_discount])
    expected_probs =  expected_weights / expected_weights.sum()
    assert np.allclose(
        buffer.get_probability(0, np.array([0, 1, 2])), # probability of last three elements
        expected_probs,
    )

    now += 2*time_increase

    # add two transitions
    transitions = make_test_transitions(0, 2, 1)
    transitions = add_ts(transitions, [now-time_increase, now])
    idxs = buffer.feed(transitions, DataMode.ONLINE)
    assert len(idxs) == 2
    # discount all previous transitions, and discount the first of the newly added transitions
    expected_weights = np.array([
        expected_discount**3,
        expected_discount**2,
        expected_discount**3,
        expected_discount,
        1,
    ])
    expected_probs =  expected_weights / expected_weights.sum()
    assert np.allclose(
        buffer.get_probability(0, np.array([0, 1, 2, 3, 4])),
        expected_probs,
    )

    # larger time increase
    now += 10*time_increase
    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now])
    idxs = buffer.feed(transitions, DataMode.ONLINE)
    assert len(idxs) == 1
    # discount all previous transitions by 10 steps, and discount the first of the newly added transitions

    expected_weights =  np.concatenate(
        (expected_weights * expected_discount**10,
        np.array([1.0]))
    )
    expected_probs =  expected_weights / expected_weights.sum()
    assert np.allclose(
        buffer.get_probability(0, np.array([0, 1, 2, 3, 4, 5])),
        expected_probs,
    )

def test_recency_bias_buffer_uniform_mixing(dummy_app_state: AppState):
    obs_period = dummy_app_state.cfg.interaction.obs_period
    buffer_cfg = RecencyBiasBufferConfig(
        seed=42,
        memory=100,
        batch_size=10,
        n_most_recent=1,
        obs_period=obs_period,
        gamma=0.9,
        ensemble=1,
        ensemble_probability=1,
        uniform_weight=0.5,
        effective_episodes=2,
    )

    buffer = RecencyBiasBuffer(buffer_cfg, dummy_app_state)
    expected_discount = np.power(0.9, 1/2)
    assert buffer._discount_factor == expected_discount
    now = datetime(2000, 1, 1)
    time_increase = timedelta(seconds=1)

    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now])
    _ = buffer.feed(transitions, DataMode.ONLINE)

    now += time_increase

    transitions = make_test_transitions(0, 1, 1)
    transitions = add_ts(transitions, [now])
    _ = buffer.feed(transitions, DataMode.ONLINE)

    expected_discounted_weights = np.array([
        expected_discount,
        1,
    ])
    expected_geometric_probs = expected_discounted_weights/expected_discounted_weights.sum()
    expected_mixed_probs = 0.5*expected_geometric_probs + 0.5*(np.ones(2)/2)
    assert np.allclose(
        buffer.get_probability(0, np.array([0, 1])),
        expected_mixed_probs,
    )
