import jax.numpy as jnp

from src.agent.components.buffer import EnsembleReplayBuffer
from src.interaction.transition_creator import Step, Transition


def test_buffer_add():
    buffer = EnsembleReplayBuffer(n_ensemble=2, max_size=10)

    step = Step(
        state=jnp.array([1.0, 2.0]),
        action=jnp.array(0.5),
        reward=1.0,
        done=False,
        gamma=0.99
    )

    transition = Transition(steps=[step], n_step_reward=1.0, n_step_gamma=0.99)
    buffer.add(transition)

    assert buffer.size == 1
    assert buffer.ptr == 1
    assert buffer.transitions[0] == transition

    mask_sum = jnp.sum(buffer.ensemble_masks[:, 0])
    assert mask_sum >= 1

def test_buffer_sample():
    buffer = EnsembleReplayBuffer(
        n_ensemble=2,
        max_size=5,
        ensemble_prob=1.0,
        seed=42
    )

    for i in range(5):
        step = Step(
            state=jnp.array([float(i), float(i)]),
            action=jnp.array(float(i)),
            reward=float(i),
            done=False,
            gamma=0.99
        )
        transition = Transition(steps=[step], n_step_reward=float(i), n_step_gamma=0.99)
        buffer.add(transition)

    batch_size = 3
    ensemble_batches = buffer.sample(batch_size)

    assert len(ensemble_batches) == 2  # n_ensemble = 2
    for batch in ensemble_batches:
        assert len(batch.steps) == batch_size

def test_buffer_overflow():
    buffer = EnsembleReplayBuffer(n_ensemble=2, max_size=2)

    for i in range(3):
        step = Step(
            state=jnp.array([float(i), float(i)]),
            action=jnp.array(float(i)),
            reward=float(i),
            done=False,
            gamma=0.99
        )
        transition = Transition(steps=[step], n_step_reward=float(i), n_step_gamma=0.99)
        buffer.add(transition)

    assert buffer.size == 2
    assert buffer.ptr == 1
    assert buffer.transitions[0] is not None
    assert buffer.transitions[0].steps[0].state[0] == 2.0

def test_ensemble_assignment():
    buffer = EnsembleReplayBuffer(
        n_ensemble=3,
        max_size=100,
        ensemble_prob=0.5,
        seed=42
    )

    for _ in range(50):
        step = Step(
            state=jnp.zeros(2),
            action=jnp.array(0.0),
            reward=0.0,
            done=False,
            gamma=0.99
        )
        transition = Transition(steps=[step], n_step_reward=0.0, n_step_gamma=0.99)
        buffer.add(transition)

    assignments_per_transition = jnp.sum(buffer.ensemble_masks[:, :buffer.size], axis=0)
    assert jnp.all(assignments_per_transition >= 1)
