from typing import NamedTuple

import jax


class JaxTransition(NamedTuple):
    last_action: jax.Array
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    gamma: jax.Array

    action_lo: jax.Array
    action_hi: jax.Array
    next_action_lo: jax.Array
    next_action_hi: jax.Array

    dp: jax.Array
    next_dp: jax.Array

    n_step_reward: jax.Array
    n_step_gamma: jax.Array

    state_dim: int
    action_dim: int
