from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import chex
import jax


@dataclass
class RandomAgentConfig:
    batch_size: int = 256


class RandomAgentState(NamedTuple):
    seed: int


class RandomAgent:
    def __init__(self, seed: int, action_dim: int):
        self.seed = seed
        self.action_dim = action_dim
        self.batch_size = 0
        self.agent_state = RandomAgentState(seed=seed)

    @partial(jax.jit, static_argnums=(0,))
    def get_actions(self, rng: chex.PRNGKey, states: jax.Array) -> jax.Array:
        batch_size = states.shape[0]
        return jax.random.uniform(rng, (batch_size, self.action_dim))

    def update(self):
        pass
