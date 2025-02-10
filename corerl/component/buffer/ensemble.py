import logging
import random
from collections.abc import Sequence
from typing import Any, Literal, cast

from discrete_dists.proportional import Proportional

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import DataMode, Transition, TransitionBatch

logger = logging.getLogger(__name__)


@config()
class EnsembleUniformReplayBufferConfig(BaseReplayBufferConfig):
    name: Literal["ensemble_uniform"] = "ensemble_uniform"
    ensemble: int = 1 # Size of the ensemble
    data_subset: float = 1.0 # Proportion of all transitions added to a given buffer in the ensemble
    seed: int = 0


class EnsembleUniformBuffer(ReplayBuffer):
    def __init__(self, cfg: EnsembleUniformReplayBufferConfig):
        super().__init__(cast(Any,  cfg))
        random.seed(self.seed)
        self.ensemble = cfg.ensemble
        self.data_subset = cfg.data_subset

        self._sub_dists = [
            Proportional(self.memory)
            for _ in range(self.ensemble)
        ]


    def feed(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().feed(transitions, data_mode)

        for dist in self._sub_dists:
            mask = self.rng.random(len(idxs)) < self.data_subset
            dist.update(idxs, mask)

        return idxs


    def load(self, transitions: Sequence[Transition], data_mode: DataMode):
        idxs = super().load(transitions, data_mode)

        for dist in self._sub_dists:
            mask = self.rng.random(len(idxs)) < self.data_subset
            dist.update(idxs, mask)

        return idxs


    def sample(self) -> list[TransitionBatch]:
        ensemble_batch: list[TransitionBatch] = []
        for dist in self._sub_dists:
            idxs = dist.sample(self.rng, self.batch_size)
            batch = self._prepare_sample(idxs)
            ensemble_batch.append(batch[0])

        return ensemble_batch

    def full_batch(self) -> list[TransitionBatch]:
        ensemble_batch: list[TransitionBatch] = []
        for _ in range(self.ensemble):
            batch = super().full_batch()[0]
            ensemble_batch.append(batch)

        return ensemble_batch

    @property
    def size(self) -> list[int]:
        return [int(d.tree.total()) for d in self._sub_dists]


buffer_group.dispatcher(EnsembleUniformBuffer)
