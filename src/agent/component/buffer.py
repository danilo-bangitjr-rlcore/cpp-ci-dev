from typing import List, NamedTuple

import jax.numpy as jnp
from jax import random

from src.interaction.transition_creator import Transition


class TransitionBatch(NamedTuple):
    steps: List[Transition]


class EnsembleReplayBuffer:
    def __init__(self, n_ensemble: int = 2, max_size: int = 1_000_000,
                 ensemble_prob: float = 0.5, seed: int = 0):
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.ptr = 0
        self.size = 0

        self.transitions: List[Transition | None] = [None] * max_size
        self.ensemble_masks = jnp.zeros((n_ensemble, max_size), dtype=bool)
        self.key = random.PRNGKey(seed)

    def add(self, transition: Transition) -> None:
        self.transitions[self.ptr] = transition

        self.key, mask_key = random.split(self.key)
        ensemble_mask = random.uniform(mask_key, (self.n_ensemble,)) < self.ensemble_prob

        if not jnp.any(ensemble_mask):
            self.key, member_key = random.split(self.key)
            random_member = random.randint(member_key, (), 0, self.n_ensemble)
            ensemble_mask = ensemble_mask.at[random_member].set(True)

        self.ensemble_masks = self.ensemble_masks.at[:, self.ptr].set(ensemble_mask)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> List[TransitionBatch]:
        ensemble_batches = []

        for i in range(self.n_ensemble):
            valid_indices = jnp.where(self.ensemble_masks[i, :self.size])[0]

            self.key, subkey = random.split(self.key)
            ind = random.choice(subkey, valid_indices, (batch_size,))

            batch_transitions = [self.transitions[idx] for idx in ind]
            batch = TransitionBatch(steps=batch_transitions)
            ensemble_batches.append(batch)

        return ensemble_batches
