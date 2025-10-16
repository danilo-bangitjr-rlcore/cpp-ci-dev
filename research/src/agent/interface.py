from typing import NamedTuple

import jax
from lib_utils.named_array import NamedArray


class Batch(NamedTuple):
    state: NamedArray
    action: jax.Array
    reward: jax.Array
    next_state: NamedArray
    gamma: jax.Array

    a_lo: jax.Array
    a_hi: jax.Array
    next_a_lo: jax.Array
    next_a_hi: jax.Array

    last_a: jax.Array
    dp: jax.Array
    next_dp: jax.Array
