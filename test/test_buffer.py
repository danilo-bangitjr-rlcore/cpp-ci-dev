import jax.numpy as jnp

from src.agent.component.buffer import EnsembleReplayBuffer, Transition


def test_buffer_add():
    buffer = EnsembleReplayBuffer(state_dim=2, action_dim=1, n_ensemble=2, max_size=10)
    state = jnp.array([1.0, 2.0])
    action = jnp.array([0.5])

    buffer.add(state, action, 1.0, state, False)

    assert buffer.size == 1
    assert buffer.ptr == 1
    assert jnp.array_equal(buffer.state[0], state)
    assert jnp.array_equal(buffer.action[0], action)
    assert jnp.array_equal(buffer.reward[0], jnp.array([1.0]))
    assert not buffer.done[0]

    mask_sum = jnp.sum(buffer.ensemble_masks[:, 0])
    assert mask_sum >= 1

def test_buffer_sample():
    buffer = EnsembleReplayBuffer(
        state_dim=2,
        action_dim=1,
        n_ensemble=2,
        max_size=5,
        ensemble_prob=1.0,
        seed=42
    )

    for i in range(5):
        state = jnp.array([float(i), float(i)])
        action = jnp.array([float(i)])
        buffer.add(state, action, float(i), state, False)

    batch_size = 3
    ensemble_batches = buffer.sample(batch_size)

    assert len(ensemble_batches) == 2  # n_ensemble = 2
    for batch in ensemble_batches:
        assert isinstance(batch, Transition)
        assert batch.state.shape == (batch_size, 2)
        assert batch.action.shape == (batch_size, 1)
        assert batch.reward.shape == (batch_size, 1)
        assert batch.next_state.shape == (batch_size, 2)
        assert batch.done.shape == (batch_size, 1)

def test_buffer_overflow():
    buffer = EnsembleReplayBuffer(state_dim=2, action_dim=1, n_ensemble=2, max_size=2)

    for i in range(3):
        state = jnp.array([float(i), float(i)])
        action = jnp.array([float(i)])
        buffer.add(state, action, float(i), state, False)

    assert buffer.size == 2
    assert buffer.ptr == 1
    assert jnp.array_equal(buffer.state[0], jnp.array([2.0, 2.0]))

def test_ensemble_assignment():
    buffer = EnsembleReplayBuffer(
        state_dim=2,
        action_dim=1,
        n_ensemble=3,
        max_size=100,
        ensemble_prob=0.5,
        seed=42
    )

    for _ in range(50):
        state = jnp.zeros(2)
        action = jnp.zeros(1)
        buffer.add(state, action, 0.0, state, False)

    # check that each transition is assigned to at least one ensemble
    assignments_per_transition = jnp.sum(buffer.ensemble_masks[:, :buffer.size], axis=0)
    assert jnp.all(assignments_per_transition >= 1)
