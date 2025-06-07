from functools import partial

import chex
import jax
import jax.numpy as jnp

import lib_utils.jax as jax_u


def test_vmap_only_levels():
    a = jnp.ones((2, 3, 4))
    b = jnp.ones(4)

    @jax_u.jit
    @partial(jax_u.vmap_only, include=["x"], levels=2)
    def f(x: jax.Array, y: jax.Array):
        chex.assert_shape(x, (4, ))
        chex.assert_shape(y, (4, ))
        return x + y

    result = f(a, b)
    assert result.shape == (2, 3, 4), f"Expected shape (2, 3, 4), got {result.shape}"
    assert jnp.all(result == 2), "Expected all elements to be 2"
