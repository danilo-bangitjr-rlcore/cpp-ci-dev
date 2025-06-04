from typing import NamedTuple

import jax
import jax.numpy as jnp

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

    batch = storage.get_batch(jnp.array([3]))
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

    batch = storage.get_ensemble_batch([jnp.array([3]), jnp.array([4])])
    assert batch.a.shape == (2, 1, 10)
    assert batch.b.shape == (2, 1)
    assert batch.c.shape == (2, 1)

    assert jnp.allclose(batch.a, jnp.array([3 * jnp.ones((1, 10)), 4 * jnp.ones((1, 10))]))
    assert jnp.allclose(batch.b, jnp.array([[3], [4]]))
    assert jnp.allclose(batch.c, jnp.array([[3.1], [4.1]]))

    assert isinstance(batch, Step)
