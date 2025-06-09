from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lib_agent.buffer.storage import ReplayStorage


def test_storage_add_floats():
    storage = ReplayStorage(10)
    idx = storage.add((1, 2, 3))
    assert idx == 0
    assert storage.size() == 1


def test_storage_add_arrays():
    storage = ReplayStorage(10)

    for i in range(10):
        idx = storage.add((i * jnp.ones(10), jnp.array(2), jnp.ones(8)))
        assert idx == i

    assert storage.last_idx() == 9


def test_storage_add_mixed():
    storage = ReplayStorage(10)

    for i in range(10):
        idx = storage.add((i * jnp.ones(10), i, i + 0.1))
        assert idx == i

    assert storage.last_idx() == 9


def test_storage_get():
    class Step(NamedTuple):
        a: jax.Array
        b: jax.Array
        c: jax.Array

    storage = ReplayStorage[Step](10)
    for i in range(10):
        storage.add(Step(i * jnp.ones(10), jnp.array(i), jnp.array(i + 0.1)))

    batch = storage.get_batch(np.array([3]))
    assert batch.a.shape == (1, 10)
    assert batch.b.shape == (1,)
    assert batch.c.shape == (1,)

    assert jnp.allclose(batch.a, 3 * jnp.ones(10))
    assert jnp.allclose(batch.b, 3)
    assert jnp.allclose(batch.c, 3.1)

    assert isinstance(batch, Step)


def test_storage_get_ensemble():
    class Step(NamedTuple):
        a: jax.Array
        b: jax.Array
        c: jax.Array

    storage = ReplayStorage[Step](10)
    for i in range(10):
        storage.add(Step(i * jnp.ones(10), jnp.array(i), jnp.array(i + 0.1)))

    batch = storage.get_ensemble_batch([np.array([3]), np.array([4])])
    assert batch.a.shape == (2, 1, 10)
    assert batch.b.shape == (2, 1)
    assert batch.c.shape == (2, 1)

    assert jnp.allclose(batch.a, jnp.array([3 * jnp.ones((1, 10)), 4 * jnp.ones((1, 10))]))
    assert jnp.allclose(batch.b, jnp.array([[3], [4]]))
    assert jnp.allclose(batch.c, jnp.array([[3.1], [4.1]]))

    assert isinstance(batch, Step)


def test_storage_last_idxs_partial():
    class Step(NamedTuple):
        a: jax.Array
        b: jax.Array
        c: jax.Array

    storage = ReplayStorage[Step](10)
    for i in range(6):
        storage.add(Step(i * jnp.ones(10), jnp.array(i), jnp.array(i + 0.1)))

    last_idxs = storage.last_idxs(4)
    assert len(last_idxs) == 4
    assert np.allclose(last_idxs, [2, 3, 4, 5])


def test_storage_last_idxs_full():
    class Step(NamedTuple):
        a: jax.Array
        b: jax.Array
        c: jax.Array

    storage = ReplayStorage[Step](10)
    for i in range(20):
        storage.add(Step(i * jnp.ones(10), jnp.array(i), jnp.array(i + 0.1)))

    last_idxs = storage.last_idxs(4)
    assert len(last_idxs) == 4
    assert np.allclose(last_idxs, [6, 7, 8, 9])

def test_storage_wraparound():
    """Test that storage correctly wraps around when capacity is exceeded."""
    storage = ReplayStorage(3)

    # Fill storage
    for i in range(3):
        idx = storage.add((i, i * 2))
        assert idx == i

    # Add one more to trigger wraparound
    idx = storage.add((10, 20))
    assert idx == 0  # Should wrap to beginning
    assert storage.size() == 3
    assert storage.last_idx() == 0


def test_storage_last_idxs_wraparound():
    """Test last_idxs when buffer has wrapped around."""
    storage = ReplayStorage(5)

    # Fill beyond capacity
    for i in range(8):
        storage.add((i, i * 2))

    # Should get indices that wrap around
    last_idxs = storage.last_idxs(3)
    assert len(last_idxs) == 3
    # pos is at 3, so last 3 indices should be [0, 1, 2]
    assert np.allclose(last_idxs, [0, 1, 2])


def test_storage_last_idxs_more_than_available():
    """Test last_idxs when requesting more than available."""
    storage = ReplayStorage(10)

    # Add only 3 items
    for i in range(3):
        storage.add((i, i * 2))

    # Request more than available
    last_idxs = storage.last_idxs(5)
    assert len(last_idxs) == 3  # Should only return what's available
    assert np.allclose(last_idxs, [0, 1, 2])
