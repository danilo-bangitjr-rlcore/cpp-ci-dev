from collections import deque
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from lib_config.config import config

from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.buffer.datatypes import DataMode


class MaskedABDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob

        self._online = Proportional(support)
        self._historical = Proportional(support)
        self._dist = MixtureDistribution(
            [
                SubDistribution(d=self._online, p=left_prob),
                SubDistribution(d=self._historical, p=1 - left_prob),
            ],
        )

    def size(self):
        return int(self._online.tree.total() + self._historical.tree.total())

    def probs(self, elements: np.ndarray) -> np.ndarray:
        return self._dist.probs(elements)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._dist.sample(rng, n)

    def update(
        self,
        rng: np.random.Generator,
        elements: np.ndarray,
        mode: DataMode,
        ensemble_mask: np.ndarray,
    ):
        batch_size = len(elements)

        is_online = mode == DataMode.ONLINE

        online_mask = np.full(batch_size, is_online)

        self._online.update(elements, ensemble_mask & online_mask)
        self._historical.update(elements, ensemble_mask & ~online_mask)


@config()
class MixedHistoryBufferConfig:
    name: Literal["mixed_history_buffer"] = "mixed_history_buffer"
    n_ensemble: int = 1
    ensemble: int = 1
    max_size: int = 1_000_000
    ensemble_probability: float = 0.5
    batch_size: int = 256
    seed: int = 0
    n_most_recent: int = 1
    online_weight: float = 0.75
    id: str = ""


class MixedHistoryBuffer[T: NamedTuple](EnsembleReplayBuffer[T]):
    def __init__(
        self,
        n_ensemble: int = 2,
        max_size: int = 1_000_000,
        ensemble_probability: float = 0.5,
        batch_size: int = 256,
        seed: int = 0,
        n_most_recent: int = 1,
        online_weight: float = 0.75,
        id: str = "",
    ):
        super().__init__(
            n_ensemble=n_ensemble,
            max_size=max_size,
            ensemble_probability=ensemble_probability,
            batch_size=batch_size,
            seed=seed,
            n_most_recent=n_most_recent,
        )

        self.online_weight = online_weight
        self.id = id

        self._ens_dists = [
            MaskedABDistribution(
                max_size,
                online_weight,
                ensemble_probability,
            ) for _ in range(n_ensemble)
        ]

        self._most_recent_online_idxs = deque(maxlen=n_most_recent)

    def _update_n_most_recent(self, idxs: np.ndarray, data_mode: DataMode) -> None:
        if data_mode == DataMode.ONLINE:
            for i in idxs:
                self._most_recent_online_idxs.appendleft(int(i))

    def _add_n_most_recent(self, idxs: np.ndarray) -> np.ndarray:
        for i, j in enumerate(self._most_recent_online_idxs):
            idxs[i] = j
        return idxs

    def feed(self, transitions: Sequence[T], data_mode: DataMode) -> np.ndarray:
        idxs = np.empty(len(transitions), dtype=np.int64)
        for j, transition in enumerate(transitions):
            idxs[j] = self._storage.add(transition)

        batch_size = len(idxs)
        ensemble_masks = self._get_ensemble_masks(batch_size)

        for dist, mask in zip(self._ens_dists, ensemble_masks, strict=True):
            dist.update(self.rng, idxs, data_mode, mask)

        self._update_n_most_recent(idxs, data_mode)

        return idxs

    def _get_ensemble_masks(self, batch_size: int) -> np.ndarray:
        ensemble_masks = self.rng.random((self.n_ensemble, batch_size)) < self.ensemble_probability

        no_ensemble = ~ensemble_masks.any(axis=0)

        for idx in np.where(no_ensemble)[0]:
            random_member = self.rng.integers(0, self.n_ensemble)
            ensemble_masks[random_member, idx] = True
        return ensemble_masks

    def sample(self):
        if not self.is_sampleable:
            raise Exception('One of the sub-distributions is empty.')

        ensemble_idxs: list[np.ndarray] = []
        for dist in self._ens_dists:
            idxs = dist.sample(self.rng, self.batch_size)
            idxs = self._add_n_most_recent(idxs)
            ensemble_idxs.append(idxs)

        return self._storage.get_ensemble_batch(ensemble_idxs)

    def get_batch(self, idxs: np.ndarray):
        return self._storage.get_batch(idxs)

    @property
    def ensemble_sizes(self) -> list[int]:
        return [d.size() for d in self._ens_dists]

    @property
    def is_sampleable(self) -> bool:
        return min(self.ensemble_sizes) > 0

def create_mixed_history_buffer_from_config(cfg: MixedHistoryBufferConfig) -> MixedHistoryBuffer:
    return MixedHistoryBuffer(
        n_ensemble=cfg.n_ensemble,
        max_size=cfg.max_size,
        ensemble_probability=cfg.ensemble_probability,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        n_most_recent=cfg.n_most_recent,
        online_weight=cfg.online_weight,
        id=cfg.id,
    )
