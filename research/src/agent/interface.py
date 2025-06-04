from typing import NamedTuple

import jax


class Batch(NamedTuple):
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    gamma: jax.Array

    a_lo: jax.Array
    a_hi: jax.Array
    next_a_lo: jax.Array
    next_a_hi: jax.Array
