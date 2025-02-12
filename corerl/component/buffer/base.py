import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group
from corerl.data_pipeline.datatypes import DataMode, StepBatch, Transition, TransitionBatch
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)


@config()
class BaseReplayBufferConfig:
    name: Any = MISSING
    seed: int = MISSING
    memory: int = 1_000_000
    batch_size: int = 256
    # Whether or not to use combined experience replay:
    #   https://arxiv.org/pdf/1712.01275
    # the number of samples in the batch from most recent data.
    n_most_recent: int = 1
    id: str = ""

class ReplayBuffer:
    def __init__(self, cfg: BaseReplayBufferConfig, app_state: AppState):
        self.seed = cfg.seed
        self.rng = np.random.default_rng(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.app_state = app_state

        self.n_most_recent = cfg.n_most_recent
        assert self.n_most_recent <= self.batch_size

        self.data = None
        self.pos = 0
        self.full = False
        self.id = cfg.id

    @abstractmethod
    def _sample_indices(self) -> np.ndarray:
        ...

    @property
    def _last_pos(self):
        return (self.pos - 1) % self.memory

    def feed(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        idxs = np.empty(len(transitions), dtype=np.int64)

        for j, transition in enumerate(transitions):
            if self.data is None:
                # Lazy instantiation
                data_size = _get_size(transition)
                self.data = [torch.empty((self.memory, *s), device=device.device) for s in data_size]

            for i, elem in enumerate(transition):
                self.data[i][self.pos] = _to_tensor(elem)

            idxs[j] = self.pos
            self.pos = (self.pos + 1) % self.memory
            if not self.full and self.pos == 0:
                self.full = True

        self.write_buffer_sizes()
        return idxs

    def write_buffer_sizes(self):
        """
        Supports writing size of multiple sub buffers
        """
        sizes = self.size
        for i, size in enumerate(sizes):
            self.app_state.metrics.write(self.app_state.agent_step, metric=f"buffer_{self.id}[{i}]_size", value=size)

    def load(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        assert len(transitions) > 0
        idxs = np.empty(len(transitions), dtype=np.int64)

        data_size = _get_size(transitions[0])
        self.data = [torch.empty((self.memory, *s)) for s in data_size]

        for idx, transition in enumerate(transitions):
            for i, elem in enumerate(transition):
                self.data[i][self.pos] = _to_tensor(elem)

            idxs[idx] = self.pos
            self.pos = (self.pos + 1) % self.memory
            if not self.full and self.pos == 0:
                self.full = True

        return idxs

    def get_batch(self, idxs:np.ndarray) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        sampled_data = [self.data[i][idxs] for i in range(len(self.data))]
        return [self._prepare(idxs, sampled_data)]

    def prepare_sample(self, idxs: np.ndarray) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        max_n_most_recent = min(*self.size, self.n_most_recent)
        for i in range(max_n_most_recent):
            idxs[i] = self._last_pos-i

        return self.get_batch(idxs)

    def sample(self) -> list[TransitionBatch]:
        idxs = self._sample_indices()
        return self.prepare_sample(idxs)

    def full_batch(self) -> list[TransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        if self.full:
            sampled_data = self.data
        else:
            sampled_data = [self.data[i][: self.pos] for i in range(len(self.data))]

        idxs = np.arange(0, len(self.data))
        return [self._prepare(idxs, sampled_data)]

    @property
    def size(self) -> list[int]:
        return [self.memory if self.full else self.pos]

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def _prepare(self, idxs: np.ndarray, batch: list[Tensor]) -> TransitionBatch:
        step_attrs = len(StepBatch.__annotations__.keys())
        prior_step_batch = StepBatch(*batch[:step_attrs])
        post_step_batch = StepBatch(*batch[step_attrs: step_attrs * 2])
        return TransitionBatch(
            idxs,
            prior_step_batch,
            post_step_batch,
            n_step_reward=batch[-2],
            n_step_gamma=batch[-1],
        )


buffer_group = Group[
    [AppState], ReplayBuffer
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
