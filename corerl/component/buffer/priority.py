import logging
from collections.abc import Sequence
from typing import Any, cast

import torch

from corerl.component.buffer.base import BaseReplayBufferConfig, ReplayBuffer, buffer_group
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import Transition, TransitionBatch

logger = logging.getLogger(__name__)


@config()
class PriorityReplayBufferConfig(BaseReplayBufferConfig):
    name: str = "priority"


class PriorityBuffer(ReplayBuffer):
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
