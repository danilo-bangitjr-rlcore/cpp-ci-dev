from typing import NamedTuple, Protocol

import jax
import numpy as np

from lib_agent.buffer.storage import ReplayStorage


class Transition(Protocol):
    @property
    def state(self) -> np.ndarray: ...

    @property
    def a_lo(self) -> np.ndarray: ...

    @property
    def a_hi(self) -> np.ndarray: ...

    @property
    def dp(self) -> bool: ...

    @property
    def last_a(self) -> np.ndarray: ...

    @property
    def action(self) -> np.ndarray: ...

    @property
    def reward(self) -> float: ...

    @property
    def gamma(self) -> float: ...

    @property
    def next_state(self) -> np.ndarray: ...

    @property
    def next_a_lo(self) -> np.ndarray: ...

    @property
    def next_a_hi(self) -> np.ndarray: ...

    @property
    def next_dp(self) -> bool: ...

    @property
    def state_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

class State(NamedTuple):
    features: jax.Array
    a_lo : jax.Array
    a_hi : jax.Array
    dp: jax.Array
    last_a: jax.Array

class EnsembleReplayBuffer[T: NamedTuple]:
    def __init__(
        self,
        n_ensemble: int = 2,
        max_size: int = 1_000_000,
        ensemble_prob: float = 0.5,
        batch_size:int = 256,
        seed: int = 0,
        n_most_recent: int = 1,
    ):
        self._storage = ReplayStorage[T](max_size)
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.batch_size = batch_size
        self.n_most_recent = n_most_recent

        self.ensemble_masks = np.zeros((n_ensemble, max_size), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def add(self, transition: T) -> None:
        ptr = self._storage.add(transition)

        ensemble_mask = self.rng.uniform(size=self.n_ensemble) < self.ensemble_prob
        # ensure that at least one member gets the transition
        if not np.any(ensemble_mask):
            random_member = self.rng.integers(0, self.n_ensemble)
            ensemble_mask[random_member] = True

        self.ensemble_masks[:, ptr] = ensemble_mask

    def sample(self):
        ens_idxs: list[np.ndarray] = []
        for m in range(self.n_ensemble):
            valid_indices = np.nonzero(self.ensemble_masks[m, :self._storage.size()])[0]

            recent_indices = np.array([])

            if self.n_most_recent > 0:
                recent_range = self._storage.last_idxs(self.n_most_recent)
                mask = self.ensemble_masks[m, recent_range]
                recent_indices = recent_range[mask]

            rand_indices = self.rng.choice(
                valid_indices,
                size=self.batch_size,
                replace=True,
            )

            combined_indices = np.concatenate([recent_indices, rand_indices])
            combined_indices = combined_indices[:self.batch_size]

            ens_idxs.append(combined_indices)
        return self._storage.get_ensemble_batch(ens_idxs)

    @property
    def size(self):
        return self._storage.size()
