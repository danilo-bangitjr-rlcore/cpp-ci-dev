from enum import Enum, auto
from typing import NamedTuple

import jax


class DataMode(Enum):
    OFFLINE = auto()
    ONLINE = auto()
    REFRESH = auto()


class Transition:
    def __init__(self, prior, post, steps, n_step_reward, n_step_gamma, state_dim, action_dim):
        self.prior = prior
        self.post = post
        self.steps = steps
        self.n_step_reward = n_step_reward
        self.n_step_gamma = n_step_gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

    @property
    def state(self):
        return self.prior.state

    @property
    def action(self):
        return self.post.action

    @property
    def reward(self):
        return self.n_step_reward

    @property
    def gamma(self):
        return self.n_step_gamma

    @property
    def next_state(self):
        return self.post.state


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
