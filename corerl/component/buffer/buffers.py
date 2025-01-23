import logging
import random
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition, TransitionBatch

logger = logging.getLogger(__name__)


@config()
class UniformReplayBufferConfig(BaseReplayBufferConfig):
    name: Literal['uniform'] = "uniform"


class UniformBuffer(ReplayBuffer):
    def __init__(self, cfg: UniformReplayBufferConfig):
        super().__init__(cfg)


    def sample(self, batch_size: int | None = None) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.randint(0, self.size, batch_size)

        if self.combined:
            sampled_indices[0] = self._last_pos

        sampled_data = [self.data[i][sampled_indices] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

buffer_group.dispatcher(UniformBuffer)


@config()
class PriorityReplayBufferConfig(BaseReplayBufferConfig):
    name: str = "priority"


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: PriorityReplayBufferConfig):
        super(PriorityBuffer, self).__init__(cast(Any, cfg))
        self.priority = torch.zeros((self.memory,))
        logger.warning("Priority buffer has not been tested yet")

    def feed(self, transitions: Sequence[Transition]) -> None:
        super(PriorityBuffer, self).feed(transitions)
        # UniformBuffer.feed() already increments self.pos. Need to take this into account
        self.pos = (self.pos - 1) % self.memory

        self.priority[self.pos] = 1.0

        self.pos += 1
        self.pos %= self.memory

        if self.full:
            scale = self.priority.sum()
        else:
            scale = self.priority[: self.pos].sum()

        self.priority /= scale

    def sample(self, batch_size: int | None = None) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.choice(
            self.size[0],
            batch_size,
            replace=False,
            p=(self.priority if self.full else self.priority[: self.pos]),
        )

        sampled_data = [self.data[i][sampled_indices] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

    def update_priorities(self, priority: torch.Tensor):
        assert priority.shape == self.priority.shape
        self.priority = torch.tensor(priority)


buffer_group.dispatcher(PriorityBuffer)


@config()
class EnsembleUniformReplayBufferConfig(BaseReplayBufferConfig):
    name: Literal["ensemble_uniform"] = "ensemble_uniform"
    ensemble: int = 10 # Size of the ensemble
    data_subset: float = 0.5 # Proportion of all transitions added to a given buffer in the ensemble


class EnsembleUniformBuffer(UniformBuffer):
    def __init__(self, cfg: EnsembleUniformReplayBufferConfig):
        super(EnsembleUniformBuffer, self).__init__(cast(Any,  cfg))
        random.seed(self.seed)
        self.ensemble = cfg.ensemble
        self.data_subset = cfg.data_subset

        sub_cfg = UniformReplayBufferConfig(
            seed=cfg.seed,
            memory=cfg.memory,
            batch_size=cfg.batch_size,
            combined=cfg.combined,
        )
        self.buffer_ensemble = [UniformBuffer(sub_cfg) for _ in range(self.ensemble)]

    def feed(self, transitions: Sequence[Transition]) -> None:
        for i in range(self.ensemble):
            if self.rng.rand() < self.data_subset:
                self.buffer_ensemble[i].feed(transitions)

    def load(self, transitions: Sequence[Transition]) -> None:
        num_transitions = len(transitions)
        assert num_transitions > 0

        subset_size = int(num_transitions * self.data_subset)

        ensemble_transitions = [random.sample(transitions, subset_size) for i in range(self.ensemble)]

        for i in range(self.ensemble):
            self.buffer_ensemble[i].load(ensemble_transitions[i])

    def sample(self, batch_size: int | None = None) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            part = self.buffer_ensemble[i].sample(batch_size)
            if part is None: continue

            ensemble_batch += part

        return ensemble_batch

    def sample_batch(self) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            part = self.buffer_ensemble[i].sample_batch()
            if part is None: continue

            ensemble_batch += part

        return ensemble_batch

    @property
    def size(self) -> list[int]:
        return [self.buffer_ensemble[i].size[0] for i in range(self.ensemble)]

    def reset(self) -> None:
        for i in range(self.ensemble):
            self.buffer_ensemble[i].reset()

buffer_group.dispatcher(EnsembleUniformBuffer)
