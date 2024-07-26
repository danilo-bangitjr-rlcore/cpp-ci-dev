from corerl.utils.device import device
from omegaconf import DictConfig
from warnings import warn
import numpy as np
import torch

from corerl.data.data import Transition, TransitionBatch


class UniformBuffer:
    def __init__(self, cfg: DictConfig):
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.data = None
        self.pos = 0
        self.full = False
        self.device = torch.device(cfg.device)

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    def feed(self, experience: Transition) -> None:
        if self.data is None:
            # Lazy instantiation
            data_size = _get_size(experience)
            self.data = tuple([
                torch.empty(
                    (self.memory, *s), device=self.device,
                ) for s in data_size
            ])

        for i, elem in enumerate(experience):
            self.data[i][self.pos] = _to_tensor(elem)

        self.pos += 1
        if not self.full and self.pos == self.memory:
            self.full = True
        self.pos %= self.memory

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        if self.size == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.randint(0, self.size, batch_size)

        sampled_data = [
            self.data[i][sampled_indices] for i in range(len(self.data))
        ]

        return [self._prepare(sampled_data)]

    def sample_batch(self) -> list[TransitionBatch]:
        if self.size == 0:
            return None

        if self.full:
            sampled_data = self.data
        else:
            sampled_data = (
                self.data[i][:self.pos] for i in range(len(self.data))
            )

        return [self._prepare(sampled_data)]

    """
    def load(
        self, states: list, actions: list, cumulants: list, dones: list,
        truncates: list,
        ) -> None:
        for i in range(len(states) - 1):
            self.feed(
                (
                    states[i], actions[i], cumulants[i], states[i+1],
                    int(dones[i]), int(truncates[i]),
                ),
            )
    """

    def load(self, transitions: list) -> None:
        for transition in transitions:
            self.feed(transition)

    @property
    def size(self) -> list[int]:
        return [self.memory if self.full else self.pos]

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def get_all_data(self) -> list:
        return self.data

    def update_priorities(self, priority=None):
        pass

    def _prepare(self, batch: list) -> TransitionBatch:
        batch = TransitionBatch(*batch)
        return batch


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: DictConfig):
        super(PriorityBuffer, self).__init__(cfg)
        self.priority = torch.zeros((self.memory,))
        warn("Priority buffer has not been tested yet")

    def feed(self, experience: Transition) -> None:
        super(PriorityBuffer, self).feed(experience)
        self.priority[self.pos] = 1.0

        if self.full:
            scale = self.priority.sum()
        else:
            scale = self.priority[:self.pos].sum()

        self.priority /= scale

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = self.rng.choice(
            self.size, batch_size, replace=False,
            p=(self.priority if self.full else self.priority[:self.pos]),
        )

        sampled_data = [
            self.data[i][sampled_indices] for i in range(len(self.data))
        ]

        return [self._prepare(sampled_data)]

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
            assert priority.shape == self.priority.shape
            self.priority = torch.Tensor(priority)


class EnsembleUniformBuffer:
    def __init__(self, cfg: DictConfig):
        print("Creating Ensemble Uniform Buffer")
        self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)
        self.batch_size = cfg.batch_size
        self.ensemble = cfg.ensemble # Size of the ensemble
        self.data_subset = cfg.data_subset # Percentage of all transitions added to a given buffer in the ensemble

        self.buffer_ensemble = [UniformBuffer(cfg) for _ in range(self.ensemble)]

        if self.batch_size == 0:
            self.sample = self.sample_batch
        else:
            self.sample = self.sample_mini_batch

    def feed(self, experience: Transition) -> None:
        for i in range(self.ensemble):
            if self.rng.rand() < self.data_subset:
                self.buffer_ensemble[i].feed(experience)

    def sample_mini_batch(self, batch_size: int = None) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            ensemble_batch += self.buffer_ensemble[i].sample_mini_batch(batch_size)

        return ensemble_batch

    def sample_batch(self) -> list[TransitionBatch]:
        ensemble_batch = []
        for i in range(self.ensemble):
            ensemble_batch += self.buffer_ensemble[i].sample_batch()

        return ensemble_batch

    def load(self, transitions: list) -> None:
        for transition in transitions:
            self.feed(transition)

    @property
    def size(self) -> list[int]:
        return [self.buffer_ensemble[i].size[0] for i in range(self.ensemble)]

    def reset(self) -> None:
        for i in range(self.ensemble):
            self.buffer_ensemble[i].reset()


def _to_tensor(elem):
    if (
        isinstance(elem, torch.Tensor) or
        isinstance(elem, np.ndarray) or
        isinstance(elem, list)
    ):
        return torch.Tensor(elem)
    else:
        return torch.Tensor([elem])


def _get_size(experience: Transition) -> list[tuple]:
    size = []
    for elem in experience:
        if isinstance(elem, np.ndarray):
            size.append(elem.shape)
        elif (
                isinstance(elem, int) or
                isinstance(elem, float) or
                isinstance(elem, bool)
        ):
            size.append((1,))
        else:
            raise TypeError(f"unknown type {type(elem)}")

    return size
