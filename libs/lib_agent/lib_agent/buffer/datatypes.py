from enum import Enum, auto
from typing import NamedTuple

import jax


class DataMode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


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

    timestamp: int | None = None

    @property
    def state_dim(self):
        return self.state.shape[-1]

    @property
    def action_dim(self):
        return self.action.shape[-1]
