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
    done: jax.Array
    gamma: jax.Array

class NPVectorizedTransition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray
    gamma: np.ndarray

    def add(self, ptr: int, transition: Transition):
        self.state[ptr, :] = transition.prior.state
        self.action[ptr, :] = transition.post.action
        self.reward[ptr, :] = transition.n_step_reward
        self.next_state[ptr, :] = transition.post.state
        self.done[ptr, :] = transition.post.done
        self.gamma[ptr, :] = transition.n_step_gamma

    def get_index(self, indices: np.ndarray):
        return NPVectorizedTransition(
            self.state[indices, :],
            self.action[indices, :],
            self.reward[indices, :],
            self.next_state[indices, :] ,
            self.done[indices, :],
            self.gamma[indices, :],
        )


def stack_transitions(transitions: list[NPVectorizedTransition]) -> VectorizedTransition:
    stacked_state = jnp.stack([t.state for t in transitions])
    stacked_action = jnp.stack([t.action for t in transitions])
    stacked_reward = jnp.stack([t.reward for t in transitions])
    stacked_next_state = jnp.stack([t.next_state for t in transitions])
    stacked_done = jnp.stack([t.done for t in transitions])
    stacked_gamma = jnp.stack([t.gamma for t in transitions])
    return VectorizedTransition(
        state=stacked_state,
        action=stacked_action,
        reward=stacked_reward,
        next_state=stacked_next_state,
        done=stacked_done,
        gamma=stacked_gamma
    )


class EnsembleReplayBuffer:
    def __init__(self, n_ensemble: int = 2, max_size: int = 1_000_000,
                 ensemble_prob: float = 0.5, batch_size:int = 256, seed: int = 0):
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size

        self.transitions = None
        self.ensemble_masks = np.zeros((n_ensemble, max_size), dtype=bool)

    def _init_transitions(self, transition: Transition) -> None:
        self.transitions = NPVectorizedTransition(
            state=np.zeros((self.max_size, transition.state_dim)),
            action=np.zeros((self.max_size,transition.action_dim)),
            reward=np.zeros((self.max_size, 1)),
            next_state=np.zeros((self.max_size, transition.state_dim)),
            done=np.zeros((self.max_size, 1)),
            gamma=np.zeros((self.max_size, 1)),
        )

    def add(self, transition: Transition) -> None:
        if self.transitions is None:
            self._init_transitions(transition)

        assert self.transitions is not None
        self.transitions.add(self.ptr, transition)

        ensemble_mask = np.random.uniform(size=self.n_ensemble) < self.ensemble_prob
        # ensure that at least one member gets the transition
        if not np.any(ensemble_mask):
            random_member = np.random.randint(0, self.n_ensemble)
            ensemble_mask[random_member] = True

        self.ensemble_masks[:, self.ptr] = ensemble_mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> VectorizedTransition:
        assert self.transitions is not None
        ensemble_samples = []
        for m in range(self.n_ensemble):
            valid_indices = np.nonzero(self.ensemble_masks[m, :self.size])[0]
            rand_indices = np.random.choice(
                valid_indices,
                size=self.batch_size,
                replace=True,
            )

            samples_m = self.transitions.get_index(rand_indices)
            ensemble_samples.append(samples_m)
        return stack_transitions(ensemble_samples)
