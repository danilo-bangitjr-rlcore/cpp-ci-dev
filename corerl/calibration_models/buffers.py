from corerl.utils.device import device
from omegaconf import DictConfig
from warnings import warn
import corerl.component.network.utils as network_utils
import numpy as np
import torch


class UniformCalibrationModelBuffer:
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

    def feed(self, experience: tuple) -> None:
        if self.data is None:
            # Lazy instatiation
            data_size = _get_size(experience)
            self.data = tuple([
                torch.empty(
                    (self.memory, *s), device=self.device,
                ) for s in data_size
            ])

        for i in range(len(experience)):
            self.data[i][self.pos] = _to_tensor(experience[i])
        self.pos = (self.pos + 1) % self.memory

        if not self.full and self.pos == self.memory:
            self.full = True

    def sample_mini_batch(self, batch_size: int = None) -> dict:
        if self.data is None or len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = self.rng.randint(
            0, len(self.data) if self.full else self.pos, batch_size,
        )

        sampled_data = (
            self.data[i][sampled_indices] for i in range(len(self.data))
        )

        return self._prepare(sampled_data)

    def sample_batch(self) -> dict:
        if len(self.data) == 0 or self.data is None:
            return None

        if self.full:
            sampled_data = self.data
        else:
            sampled_data = (
                self.data[i][:self.pos] for i in range(len(self.data))
            )

        return self._prepare(sampled_data)

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
    def size(self) -> int:
        return len(self.data)

    def reset(self) -> None:
        self.data = []
        self.pos = 0

    def get_all_data(self) -> list:
        return self.data

    def update_priorities(self, priority=None):
        pass

    def _prepare(self, batch: list) -> dict:
        s, a, r, ns, d, t, dp, ndp, ge = batch
        return {
            'states': s,
            'actions': a,
            'rewards': r,
            'next_states': ns,
            'dones': d,
            'truncs': t,
            'state_decision_points': dp,
            'next_state_decision_points': ndp,
            'gamma_exps': ge
        }


class PriorityBuffer(UniformBuffer):
    def __init__(self, cfg: DictConfig):
        super(PriorityBuffer, self).__init__(cfg)
        self.priority = torch.zeros((self.memory,))
        warn("Priority buffer has not been tested yet")

    def feed(self, experience: tuple) -> None:
        super(PriorityBuffer, self).feed(experience)
        self.priority[self.pos] = 1.0

        if self.full:
            scale = self.priority.sum()
        else:
            scale = self.priority[:self.pos].sum()

        self.priority /= scale

    def sample_mini_batch(self, batch_size: int = None) -> dict:
        if len(self.data) == 0:
            return None
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = self.rng.choice(
            self.size, batch_size, replace=False,
            p=(self.priority if self.full else self.priority[:self.pos]),
        )

        sampled_data = (
            self.data[i][sampled_indices] for i in range(len(self.data))
        )

        return self._prepare(sampled_data)

    def update_priorities(self, priority=None):
        if priority is None:
            raise NotImplementedError
        else:
            assert priority.shape == self.priority.shape
            self.priority = torch.Tensor(priority)


def _to_tensor(elem):
    if (
        isinstance(elem, torch.Tensor) or
        isinstance(elem, np.ndarray) or
        isinstance(elem, list)
    ):
        return torch.Tensor(elem)
    else:
        return torch.Tensor([elem])


def _get_size(experience: tuple) -> int:
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
