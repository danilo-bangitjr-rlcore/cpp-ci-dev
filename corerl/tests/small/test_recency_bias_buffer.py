from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from lib_agent.buffer.datatypes import DataMode
from lib_agent.buffer.factory import build_buffer
from lib_agent.buffer.recency_bias_buffer import RecencyBiasBuffer, RecencyBiasBufferConfig


def test_recency_bias_buffer_add_with_timestamps():
    """
    Verifies that add() with timestamps properly biases toward recent transitions.

    The recency bias mechanism works by discounting ALL existing transitions
    when a new transition is added, based on elapsed time since last add.
    """
    cfg = RecencyBiasBufferConfig(
        obs_period=1,
        gamma=[0.5],
        effective_episodes=[10],
        ensemble=1,
        uniform_weight=0.1,
        ensemble_probability=1.0,
        max_size=1000,
        batch_size=32,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        ensemble=cfg.ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
        batch_size=cfg.batch_size,
    )

    class TransitionWithTimestamp(NamedTuple):
        idx: int
        last_action: jnp.ndarray
        state: jnp.ndarray
        action: jnp.ndarray
        reward: jnp.ndarray
        next_state: jnp.ndarray
        gamma: jnp.ndarray
        action_lo: jnp.ndarray
        action_hi: jnp.ndarray
        next_action_lo: jnp.ndarray
        next_action_hi: jnp.ndarray
        dp: jnp.ndarray
        next_dp: jnp.ndarray
        n_step_reward: jnp.ndarray
        n_step_gamma: jnp.ndarray
        timestamp: int

    timestamps = [0, 5, 10]

    for i, ts in enumerate(timestamps):
        transition = TransitionWithTimestamp(
            idx=i,
            last_action=jnp.array([i]),
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=jnp.array([i]),
            next_state=jnp.array([i + 1]),
            gamma=jnp.array([0.99]),
            action_lo=jnp.array([0.0]),
            action_hi=jnp.array([1.0]),
            next_action_lo=jnp.array([0.0]),
            next_action_hi=jnp.array([1.0]),
            dp=jnp.array([True]),
            next_dp=jnp.array([True]),
            n_step_reward=jnp.array([i]),
            n_step_gamma=jnp.array([0.99]),
            timestamp=ts,
        )
        buffer.add(transition)

    assert buffer.size == 3

    probs = buffer.get_probability(0, np.array([0, 1, 2]))

    assert probs[2] > probs[1] > probs[0], "More recent transitions should have higher probability"

    prob_ratio = probs[2] / probs[0]
    assert prob_ratio > 1.5, f"Expected significant recency bias, got ratio {prob_ratio}"


def test_recency_bias_buffer_feed_with_timestamps():
    """
    Test that feed() method respects timestamps when provided.
    """
    cfg = RecencyBiasBufferConfig(
        obs_period=1,
        gamma=[0.5],
        effective_episodes=[10],
        ensemble=1,
        uniform_weight=0.1,
        ensemble_probability=1.0,
        max_size=1000,
        batch_size=32,
    )
    buffer = RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=cfg.gamma,
        effective_episodes=cfg.effective_episodes,
        ensemble=cfg.ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        max_size=cfg.max_size,
        batch_size=cfg.batch_size,
    )

    class TransitionWithTimestamp(NamedTuple):
        idx: int
        last_action: jnp.ndarray
        state: jnp.ndarray
        action: jnp.ndarray
        reward: jnp.ndarray
        next_state: jnp.ndarray
        gamma: jnp.ndarray
        action_lo: jnp.ndarray
        action_hi: jnp.ndarray
        next_action_lo: jnp.ndarray
        next_action_hi: jnp.ndarray
        dp: jnp.ndarray
        next_dp: jnp.ndarray
        n_step_reward: jnp.ndarray
        n_step_gamma: jnp.ndarray
        timestamp: int

        @property
        def state_dim(self):
            return 1

        @property
        def action_dim(self):
            return 1

    timestamps = [0, 5, 10]
    transitions = [
        TransitionWithTimestamp(
            idx=i,
            last_action=jnp.array([i]),
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=jnp.array([i]),
            next_state=jnp.array([i + 1]),
            gamma=jnp.array([0.99]),
            action_lo=jnp.array([0.0]),
            action_hi=jnp.array([1.0]),
            next_action_lo=jnp.array([0.0]),
            next_action_hi=jnp.array([1.0]),
            dp=jnp.array([True]),
            next_dp=jnp.array([True]),
            n_step_reward=jnp.array([i]),
            n_step_gamma=jnp.array([0.99]),
            timestamp=timestamps[i],
        )
        for i in range(3)
    ]

    buffer.feed(transitions, DataMode.ONLINE)

    assert buffer.size == 3

    probs = buffer.get_probability(0, np.array([0, 1, 2]))

    assert probs[2] > probs[1] > probs[0], "More recent transitions should have higher probability"

    prob_ratio = probs[2] / probs[0]
    assert prob_ratio > 1.5, f"Expected significant recency bias, got ratio {prob_ratio}"


def test_recency_bias_buffer_factory():
    """
    Verifies that buffer can be created via factory function.
    """
    class TransitionWithTimestamp(NamedTuple):
        idx: int
        last_action: jnp.ndarray
        state: jnp.ndarray
        action: jnp.ndarray
        reward: jnp.ndarray
        next_state: jnp.ndarray
        gamma: jnp.ndarray
        action_lo: jnp.ndarray
        action_hi: jnp.ndarray
        next_action_lo: jnp.ndarray
        next_action_hi: jnp.ndarray
        dp: jnp.ndarray
        next_dp: jnp.ndarray
        n_step_reward: jnp.ndarray
        n_step_gamma: jnp.ndarray
        timestamp: int

    cfg = RecencyBiasBufferConfig(
        obs_period=1000,
        gamma=[0.99, 0.99],
        effective_episodes=[100, 100],
        ensemble=2,
        uniform_weight=0.1,
        ensemble_probability=0.5,
        max_size=1000,
        batch_size=32,
    )

    buffer = build_buffer(cfg, TransitionWithTimestamp)

    assert isinstance(buffer, RecencyBiasBuffer)
    assert buffer.ensemble == 2
    assert buffer.max_size == 1000
