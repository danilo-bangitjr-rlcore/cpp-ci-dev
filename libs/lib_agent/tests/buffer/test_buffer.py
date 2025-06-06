from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lib_agent.buffer.buffer import EnsembleReplayBuffer


class FakeTransition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: float
    next_state: jnp.ndarray
    done: bool


def test_buffer_add_single():
    buffer = EnsembleReplayBuffer[FakeTransition](n_ensemble=2, max_size=10, seed=42)

    transition = FakeTransition(
        state=jnp.array([1.0, 2.0]),
        action=jnp.array([0.5]),
        reward=1.0,
        next_state=jnp.array([3.0, 4.0]),
        done=False,
    )

    buffer.add(transition)
    assert buffer.size == 1

    # At least one ensemble member should have this transition
    assert np.any(buffer.ensemble_masks[:, 0])


def test_buffer_add_multiple():
    buffer = EnsembleReplayBuffer[FakeTransition](n_ensemble=2, max_size=10, seed=42)

    for i in range(5):
        transition = FakeTransition(
            state=jnp.array([i, i+1]),
            action=jnp.array([i*0.1]),
            reward=float(i),
            next_state=jnp.array([i+2, i+3]),
            done=i == 4,
        )
        buffer.add(transition)

    assert buffer.size == 5


def test_buffer_ensemble_mask_at_least_one():
    """Test that at least one ensemble member gets each transition."""
    buffer = EnsembleReplayBuffer[FakeTransition](
        n_ensemble=3,
        max_size=10,
        ensemble_prob=0.0,
        seed=42,
    )

    transition = FakeTransition(
        state=jnp.array([1.0]),
        action=jnp.array([0.5]),
        reward=1.0,
        next_state=jnp.array([2.0]),
        done=False,
    )

    buffer.add(transition)

    # Even with 0 probability, at least one member should get it
    assert np.sum(buffer.ensemble_masks[:, 0]) >= 1


def test_buffer_sample_basic():
    buffer = EnsembleReplayBuffer[FakeTransition](
        n_ensemble=2,
        max_size=100,
        batch_size=4,
        seed=42,
    )

    # Add some transitions
    for i in range(10):
        transition = FakeTransition(
            state=jnp.array([i, i+1]),
            action=jnp.array([i*0.1]),
            reward=float(i),
            next_state=jnp.array([i+2, i+3]),
            done=False,
        )
        buffer.add(transition)

    batch = buffer.sample()

    # Should return batch for each ensemble member
    assert batch.state.shape == (2, 4, 2)  # (n_ensemble, batch_size, state_dim)
    assert batch.action.shape == (2, 4, 1)
    assert jnp.asarray(batch.reward).shape == (2, 4)
    assert batch.next_state.shape == (2, 4, 2)
    assert jnp.asarray(batch.done).shape == (2, 4)

    assert isinstance(batch, FakeTransition)


def test_buffer_sample_with_recent():
    buffer = EnsembleReplayBuffer[FakeTransition](
        n_ensemble=1,
        max_size=100,
        batch_size=4,
        n_most_recent=2,
        seed=42,
    )

    # Add transitions
    for i in range(10):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition)

    batch = buffer.sample()
    assert batch.state.shape == (1, 4, 1)
    assert jnp.all(
        jnp.asarray(batch.reward)[:, :2] == jnp.array([[8.0, 9.0]]),
    )


def test_buffer_wraparound():
    """Test buffer behavior when it wraps around."""
    buffer = EnsembleReplayBuffer[FakeTransition](n_ensemble=2, max_size=3, seed=42)

    # Fill beyond capacity
    for i in range(5):
        transition = FakeTransition(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=float(i),
            next_state=jnp.array([i+1]),
            done=False,
        )
        buffer.add(transition)

    assert buffer.size == 3  # Should be capped at max_size

    # Ensemble masks should have wrapped around too
    assert buffer.ensemble_masks.shape == (2, 3)


def test_buffer_ensemble_probability():
    """Test that ensemble probability affects mask generation."""
    # High probability - most transitions should be in multiple ensembles
    buffer_high = EnsembleReplayBuffer[FakeTransition](
        n_ensemble=3,
        max_size=10,
        ensemble_prob=0.9,
        seed=42,
    )

    # Low probability - fewer transitions in multiple ensembles
    buffer_low = EnsembleReplayBuffer[FakeTransition](
        n_ensemble=3,
        max_size=10,
        ensemble_prob=0.1,
        seed=42,
    )

    transition = FakeTransition(
        state=jnp.array([1.0]),
        action=jnp.array([0.5]),
        reward=1.0,
        next_state=jnp.array([2.0]),
        done=False,
    )

    # Add same transition to both buffers
    buffer_high.add(transition)
    buffer_low.add(transition)

    high_count = np.sum(buffer_high.ensemble_masks[:, 0])
    low_count = np.sum(buffer_low.ensemble_masks[:, 0])

    # Both should have at least 1, but high prob should generally have more
    assert high_count >= 1
    assert low_count >= 1
    assert high_count > low_count
