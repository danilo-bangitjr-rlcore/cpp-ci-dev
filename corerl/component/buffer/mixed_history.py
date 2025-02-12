from collections.abc import Sequence
from typing import Literal

import numpy as np
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from numpy.random._generator import Generator as Generator

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import DataMode, Transition, TransitionBatch


@config()
class MixedHistoryBufferConfig(BaseReplayBufferConfig):
    name: Literal["mixed_history_buffer"] = "mixed_history_buffer"

    online_weight: float = 0.75
    ensemble: int = 10
    ensemble_probability: float = 0.5


class MixedHistoryBuffer(ReplayBuffer):
    def __init__(self, cfg: MixedHistoryBufferConfig):
        super().__init__(cfg)
        self._cfg = cfg

        self._sub_dists = [
            MaskedABDistribution(self.memory, cfg.online_weight, cfg.ensemble_probability)
            for _ in range(cfg.ensemble)
        ]


    def feed(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().feed(transitions, data_mode)

        for dist in self._sub_dists:
            dist.update(self.rng, idxs, data_mode)

        return idxs


    def load(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().load(transitions, data_mode)

        for dist in self._sub_dists:
            dist.update(self.rng, idxs, data_mode)

        return idxs

    def sample(self) -> list[TransitionBatch]:
        ensemble_batch: list[TransitionBatch] = []
        for dist in self._sub_dists:
            idxs = dist.sample(self.rng, self.batch_size)
            batch = self.prepare_sample(idxs)
            ensemble_batch.append(batch[0])

        return ensemble_batch


    def full_batch(self) -> list[TransitionBatch]:
        ensemble_batch: list[TransitionBatch] = []
        for _ in range(self._cfg.ensemble):
            batch = super().full_batch()[0]
            ensemble_batch.append(batch)

        return ensemble_batch

    @property
    def size(self) -> list[int]:
        return [d.size() for d in self._sub_dists]


buffer_group.dispatcher(MixedHistoryBuffer)


class MaskedABDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob

        self._online = Proportional(support)
        self._historical = Proportional(support)
        self._dist = MixtureDistribution([
            SubDistribution(d=self._online, p=left_prob),
            SubDistribution(d=self._historical, p=1-left_prob),
        ])

    def size(self):
        # define the number of elements in this buffer
        # as the total number of non-zero elements in either
        # distribution --- represented by the sum of the sumtree
        # since elements are either 1 or 0
        return int(
            self._online.tree.total() + \
            self._historical.tree.total()
        )

    def probs(self, elements: np.ndarray) -> np.ndarray:
        return self._dist.probs(elements)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._dist.sample(rng, n)

    def update(self, rng: np.random.Generator, elements: np.ndarray, mode: DataMode):
        batch_size = len(elements)

        ensemble_mask = rng.random(batch_size) < self._mask_prob
        online_mask = np.full(batch_size, mode == DataMode.ONLINE)

        self._online.update(elements, ensemble_mask & online_mask)
        self._historical.update(elements, ensemble_mask & ~online_mask)
