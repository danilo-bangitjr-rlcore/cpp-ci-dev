from typing import Any, Literal, cast
import torch
import random
import logging
import numpy as np

from torch import Tensor
from corerl.configs.config import config
from corerl.utils.device import device
from corerl.data_pipeline.datatypes import NewTransition, NewTransitionBatch, StepBatch
from corerl.configs.group import Group
from corerl.configs.config import MISSING

logger = logging.getLogger(__name__)


@config()
class BaseReplayBufferConfig:
    name: Any = MISSING
    seed: int = MISSING
    memory: int = 1000000
    batch_size: int = 256
    combined: bool = True


@config()
class UniformReplayBufferConfig(BaseReplayBufferConfig):
    name: Literal['uniform'] = "uniform"


class UniformBuffer:
    def __init__(self, cfg: UniformReplayBufferConfig):
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

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    @property
    def _last_pos(self):
        if self.pos == 0 and not self.full:
            return 0
        else:
            return self.pos - 1

    def feed(self, experience: NewTransition) -> None:
        if self.data is None:
            # Lazy instantiation
            data_size = _get_size(experience)
            self.data = [torch.empty((self.memory, *s), device=device.device) for s in data_size]

        for i, elem in enumerate(experience):
            self.data[i][self.pos] = _to_tensor(elem)

        self.pos += 1
        if not self.full and self.pos == self.memory:
            self.full = True
        self.pos %= self.memory

    def load(self, transitions: list[NewTransition]) -> None:
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

    def sample_mini_batch(self, batch_size: int | None = None) -> list[NewTransitionBatch]:
        if self.size == [0] or self.data is None:
            return []

        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.randint(0, self.size, batch_size)

        if self.combined:
            sampled_indices[0] = self._last_pos

        sampled_data = [self.data[i][sampled_indices] for i in range(len(self.data))]

        return [self._prepare(sampled_data)]

    def sample_batch(self) -> list[NewTransitionBatch]:
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

    def get_all_data(self) -> list | None:
        return self.data

    def update_priorities(self, priority=None):
        pass

    def _prepare(self, batch: list[Tensor]) -> NewTransitionBatch:
        step_attrs = len(StepBatch.__annotations__.keys())
        prior_step_batch = StepBatch(*batch[:step_attrs])
        post_step_batch = StepBatch(*batch[step_attrs: step_attrs * 2])
        return NewTransitionBatch(
            prior_step_batch,
            post_step_batch,
            n_step_reward=batch[-2],
            n_step_gamma=batch[-1])


buffer_group = Group[
    [], UniformBuffer
]()

buffer_group.dispatcher(UniformBuffer)


@config()
class PriorityReplayBufferConfig(BaseReplayBufferConfig):
    name: str = "priority"


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: PriorityReplayBufferConfig):
        super(PriorityBuffer, self).__init__(cast(Any, cfg))
        self.priority = torch.zeros((self.memory,))
        logger.warning("Priority buffer has not been tested yet")

    def feed(self, experience: NewTransition) -> None:
        super(PriorityBuffer, self).feed(experience)
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

    def sample_mini_batch(self, batch_size: int | None = None) -> list[NewTransitionBatch]:
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

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
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

    def feed(self, experience: NewTransition) -> None:
        for i in range(self.ensemble):
            if self.rng.rand() < self.data_subset:
                self.buffer_ensemble[i].feed(experience)

    def load(self, transitions: list[NewTransition]) -> None:
        num_transitions = len(transitions)
        assert num_transitions > 0

        subset_size = int(num_transitions * self.data_subset)

        ensemble_transitions = [random.sample(transitions, subset_size) for i in range(self.ensemble)]

        for i in range(self.ensemble):
            self.buffer_ensemble[i].load(ensemble_transitions[i])

    def sample_mini_batch(self, batch_size: int | None = None) -> list[NewTransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            part = self.buffer_ensemble[i].sample_mini_batch(batch_size)
            if part is None: continue

            ensemble_batch += part

        return ensemble_batch

    def sample_batch(self) -> list[NewTransitionBatch]:
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

    def subsampling(self, idxs: list[list[int]]) -> None:
        for i in range(self.ensemble):
            all_data = self.buffer_ensemble[i].data
            if all_data is None:
                continue
            idx = list(set(idxs[i]))
            new_data = []
            for attr in range(len(all_data)):
                new_data_attr = torch.empty(all_data[attr].size(),
                                            device=device.device)
                new_data_attr[np.arange(len(idx))] = all_data[attr][idx]
                new_data.append(new_data_attr)
            self.buffer_ensemble[i].pos = len(idx)
            self.buffer_ensemble[i].full = False if len(idx) != self.buffer_ensemble[i].memory else True
            self.buffer_ensemble[i].data = new_data


buffer_group.dispatcher(EnsembleUniformBuffer)


def _to_tensor(elem):
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


def _get_size(experience: NewTransition) -> list[tuple]:
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
