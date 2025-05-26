from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from interaction.transition_creator import Transition


class VectorizedTransition(NamedTuple):
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    gamma: jax.Array

class NPVectorizedTransition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    gamma: np.ndarray

    def add(self, ptr: int, transition: Transition):
        self.state[ptr, :] = transition.state
        self.action[ptr, :] = transition.action
        self.reward[ptr, :] = transition.reward
        self.next_state[ptr, :] = transition.next_state
        self.gamma[ptr, :] = transition.gamma

    def get_index(self, indices: np.ndarray):
        return NPVectorizedTransition(
            self.state[indices, :],
            self.action[indices, :],
            self.reward[indices, :],
            self.next_state[indices, :] ,
            self.gamma[indices, :],
        )


def stack_transitions(transitions: list[NPVectorizedTransition]) -> VectorizedTransition:
    stacked_state = jnp.stack([t.state for t in transitions])
    stacked_action = jnp.stack([t.action for t in transitions])
    stacked_reward = jnp.stack([t.reward for t in transitions])
    stacked_next_state = jnp.stack([t.next_state for t in transitions])
    stacked_gamma = jnp.stack([t.gamma for t in transitions])
    return VectorizedTransition(
        state=stacked_state,
        action=stacked_action,
        reward=stacked_reward,
        next_state=stacked_next_state,
        gamma=stacked_gamma
    )


class EnsembleReplayBuffer:
    def __init__(self, n_ensemble: int = 2, max_size: int = 1_000_000,
                 ensemble_prob: float = 0.5, batch_size:int = 256, seed: int = 0,
                 n_most_recent: int = 1):
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size
        self.n_most_recent = n_most_recent

        self.transitions = None
        self.ensemble_masks = np.zeros((n_ensemble, max_size), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def _init_transitions(self, transition: Transition) -> None:
        self.transitions = NPVectorizedTransition(
            state=np.zeros((self.max_size, transition.state_dim)),
            action=np.zeros((self.max_size,transition.action_dim)),
            reward=np.zeros((self.max_size, 1)),
            next_state=np.zeros((self.max_size, transition.state_dim)),
            gamma=np.zeros((self.max_size, 1)),
        )

    def add(self, transition: Transition) -> None:
        if self.transitions is None:
            self._init_transitions(transition)

        assert self.transitions is not None
        self.transitions.add(self.ptr, transition)

        ensemble_mask = self.rng.uniform(size=self.n_ensemble) < self.ensemble_prob
        # ensure that at least one member gets the transition
        if not np.any(ensemble_mask):
            random_member = self.rng.integers(0, self.n_ensemble)
            ensemble_mask[random_member] = True

        self.ensemble_masks[:, self.ptr] = ensemble_mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> VectorizedTransition:
        assert self.transitions is not None
        ensemble_samples = []
        for m in range(self.n_ensemble):
            valid_indices = np.nonzero(self.ensemble_masks[m, :self.size])[0]

            n_random = self.batch_size
            recent_indices = np.array([])

            # if n_most_recent is set, include the most recent transitions
            if self.n_most_recent > 0:
                # get the most recent indices that are valid for this ensemble member
                if self.size >= self.max_size:
                    # buffer is full, recent indices wrap around
                    recent_range = np.arange(self.ptr, self.ptr + self.n_most_recent) % self.max_size
                else:
                    # buffer is not full, recent indices are at the end
                    start_idx = max(0, self.size - self.n_most_recent)
                    recent_range = np.arange(start_idx, self.size)

                # filter for valid indices for this ensemble member
                mask = self.ensemble_masks[m, recent_range]
                recent_valid = recent_range[mask]
                if len(recent_valid) > 0:
                    recent_indices = recent_valid
                n_random = max(0, self.batch_size - len(recent_indices))

            if n_random > 0:
                # remove recent indices from valid_indices to avoid duplicates
                if recent_indices.size > 0:
                    valid_for_random = np.setdiff1d(valid_indices, recent_indices)
                    # if we don't have enough valid indices after removing recent ones, allow duplicates
                    if len(valid_for_random) < n_random:
                        valid_for_random = valid_indices
                else:
                    valid_for_random = valid_indices

                rand_indices = self.rng.choice(
                    valid_for_random,
                    size=n_random,
                    replace=True,
                )

                # combine recent and random indices
                combined_indices = np.concatenate([recent_indices, rand_indices])
            else:
                # if we have enough or too many recent indices, sample from them
                combined_indices = self.rng.choice(
                    recent_indices,
                    size=self.batch_size,
                    replace=True,
                )

            samples_m = self.transitions.get_index(combined_indices)
            ensemble_samples.append(samples_m)
        return stack_transitions(ensemble_samples)
