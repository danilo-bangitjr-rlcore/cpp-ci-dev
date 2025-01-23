import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import StepBatch, Transition, TransitionBatch
from corerl.utils.device import device

logger = logging.getLogger(__name__)


@config()
class BaseReplayBufferConfig:
    name: Any = MISSING
    seed: int = MISSING
    memory: int = 1_000_000
    batch_size: int = 256
    combined: bool = True


class ReplayBuffer:
    def __init__(self, cfg: BaseReplayBufferConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size

        # Whether or not to use combined experience replay:
        #   https://arxiv.org/pdf/1712.01275
        self.combined = cfg.combined

        self.data = None
        self.pos = 0
        self.full = False


    @property
    def _last_pos(self):
        return (self.pos - 1) % self.memory

    def feed(self, transitions: Sequence[Transition]) -> None:
        for transition in transitions:
            if self.data is None:
                # Lazy instantiation
                data_size = _get_size(transition)
                self.data = [torch.empty((self.memory, *s), device=device.device) for s in data_size]

            for i, elem in enumerate(transition):
                self.data[i][self.pos] = _to_tensor(elem)

            self.pos += 1
            if not self.full and self.pos == self.memory:
                self.full = True
            self.pos %= self.memory

    def load(self, transitions: Sequence[Transition]) -> None:
        assert len(transitions) > 0

        data_size = _get_size(transitions[0])
        self.data = [torch.empty((self.memory, *s)) for s in data_size]

        for transition in transitions:
            for i, elem in enumerate(transition):
                self.data[i][self.pos] = _to_tensor(elem)

            self.pos += 1
            if not self.full and self.pos == self.memory:
                self.full = True
            self.pos %= self.memory

        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(device.device)

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

    def sample_batch(self) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        if self.full:
            sampled_data = self.data
        else:
            sampled_data = [self.data[i][: self.pos] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

    @property
    def size(self) -> list[int]:
        return [self.memory if self.full else self.pos]

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def _prepare(self, batch: list[Tensor]) -> TransitionBatch:
        step_attrs = len(StepBatch.__annotations__.keys())
        prior_step_batch = StepBatch(*batch[:step_attrs])
        post_step_batch = StepBatch(*batch[step_attrs: step_attrs * 2])
        return TransitionBatch(
            prior_step_batch,
            post_step_batch,
            n_step_reward=batch[-2],
            n_step_gamma=batch[-1])


buffer_group = Group[
    [], ReplayBuffer
]()


def _to_tensor(elem: object):
    if (
            isinstance(elem, Tensor)
            or isinstance(elem, np.ndarray)
            or isinstance(elem, list)
    ):
        return Tensor(elem)
    elif elem is None:
        return torch.empty((1, 0))
    else:
        return Tensor([elem])


def _get_size(experience: Transition) -> list[tuple]:
    size = []
    for elem in experience:
        if isinstance(elem, np.ndarray):
            size.append(elem.shape)
        elif isinstance(elem, Tensor):
            size.append(tuple(elem.shape))
        elif elem is None:
            size.append((0,))
        elif isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, bool):
            size.append((1,))
        elif isinstance(elem, list):
            size.append((len(elem),))
        else:
            raise TypeError(f"unknown type {type(elem)}")

    return size
