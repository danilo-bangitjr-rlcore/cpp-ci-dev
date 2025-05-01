from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.datatypes import DataMode, StepBatch, Transition, TransitionBatch
from corerl.state import AppState
from corerl.utils.device import device

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class MixedHistoryBufferConfig:
    name: Literal["mixed_history_buffer"] = "mixed_history_buffer"
    online_weight: float = 0.75
    ensemble: int = MISSING
    ensemble_probability: float = 0.5
    seed: int = MISSING
    memory: int = 1_000_000
    batch_size: int = 256
    # Whether or not to use combined experience replay:
    #   https://arxiv.org/pdf/1712.01275
    # the number of samples in the batch from most recent data.
    n_most_recent: int = 1
    id: str = ""

    @computed("seed")
    @classmethod
    def _seed(cls, cfg: "MainConfig"):
        return cfg.seed

    @computed("ensemble")
    @classmethod
    def _ensemble(cls, cfg: "MainConfig"):
        return cfg.feature_flags.ensemble

class MixedHistoryBuffer:
    def __init__(self, cfg: MixedHistoryBufferConfig, app_state: AppState):
        self._cfg = cfg

        self.seed = cfg.seed
        self.rng = np.random.default_rng(self.seed)
        self.memory = cfg.memory
        self.batch_size = cfg.batch_size
        self.app_state = app_state

        assert cfg.n_most_recent <= self.batch_size
        self.n_most_recent = cfg.n_most_recent

        self.data = None
        self.pos = 0
        self.full = False
        self.id = cfg.id

        self._sub_dists = [
            MaskedABDistribution(self.memory, cfg.online_weight, cfg.ensemble_probability) for _ in range(cfg.ensemble)
        ]
        self._most_recent_online_idxs = deque(maxlen=cfg.n_most_recent)

    def feed(self, transitions: Sequence[Transition], data_mode: DataMode) -> np.ndarray:
        """
        Adds transitions to the buffer.
        """
        idxs = np.empty(len(transitions), dtype=np.int64)

        for j, transition in enumerate(transitions):
            if self.data is None:
                # Lazy instantiation
                data_size = _get_size(transition)
                self.data = [torch.empty((self.memory, *s), device=device.device) for s in data_size]

            i = 0
            for elem in transition:
                self.data[i][self.pos] = _to_tensor(elem)
                i += 1

            idxs[j] = self.pos
            self.pos = (self.pos + 1) % self.memory
            if not self.full and self.pos == 0:
                self.full = True

        batch_size = len(idxs)

        # generate a random mask for each ensemble member
        ensemble_masks = self.rng.random((len(self._sub_dists), batch_size)) < self._cfg.ensemble_probability

        # for any data point not selected by any ensemble member, randomly select one member
        no_ensemble = ~ensemble_masks.any(axis=0)

        for idx in np.where(no_ensemble)[0]:
            random_member = self.rng.integers(0, len(self._sub_dists))
            ensemble_masks[random_member, idx] = True

        for dist, mask in zip(self._sub_dists, ensemble_masks, strict=False):
            dist.update(self.rng, idxs, data_mode, mask)

        self.write_buffer_sizes()

        # update the most_recent_idxs
        if data_mode == DataMode.ONLINE:
            for i in idxs:
                self._most_recent_online_idxs.appendleft(int(i))

        return idxs

    def sample(self) -> list[TransitionBatch]:
        """
        Samples a list of TransitionBatch from the buffer.

        Raises an exception when one of the sub-distributions is empty.
        """
        if not self.is_sampleable:
            raise Exception('One of the sub-distributions is empty.')

        ensemble_batch: list[TransitionBatch] = []
        for dist in self._sub_dists:
            idxs = dist.sample(self.rng, self.batch_size)
            idxs = self._add_n_most_recent(idxs)
            batch = self.get_batch(idxs)
            ensemble_batch.append(batch)

        return ensemble_batch

    def _add_n_most_recent(self, idxs: np.ndarray) -> np.ndarray:
        """
        Iterates over the sampled idxs and adds the n most recent online idxs to the beginning of the list.
        """
        for i, j in enumerate(self._most_recent_online_idxs):
            idxs[i] = j
        return idxs

    def get_batch(self, idxs: np.ndarray) -> TransitionBatch:
        """
        Given an array of indices, returns a TransitionBatch where the entries are the transitions
        at the given indices.
        """
        assert self.data is not None
        sampled_data = [self.data[i][idxs] for i in range(len(self.data))]
        return self._prepare_batch(idxs, sampled_data)

    def _prepare_batch(self, idxs: np.ndarray, batch: list[torch.Tensor]) -> TransitionBatch:
        """
        Given an array of indices and a list of tensors representing the raw data of transitions,
        returns a TransitionBatch.
        """
        step_attrs = len(StepBatch.__annotations__.keys())
        prior_step_batch = StepBatch(*batch[:step_attrs])
        post_step_batch = StepBatch(*batch[step_attrs : step_attrs * 2])
        return TransitionBatch(
            idxs,
            prior_step_batch,
            post_step_batch,
            n_step_reward=batch[-2],
            n_step_gamma=batch[-1],
        )

    @property
    def size(self) -> list[int]:
        """
        Size of each sub-distribution.
        """
        return [d.size() for d in self._sub_dists]

    @property
    def is_sampleable(self) -> bool:
        """
        Checks to see whether the buffer is ready to be sampled
        """
        return min(self.size) > 0

    def reset(self) -> None:
        """
        Resets the buffer to its original state.
        """
        self.data = None
        self.pos = 0
        self.full = False

    def write_buffer_sizes(self):
        """
        Write the sizes of the sub buffers to metrics.
        """
        sizes = self.size
        for i, size in enumerate(sizes):
            self.app_state.metrics.write(self.app_state.agent_step, metric=f"buffer_{self.id}[{i}]_size", value=size)


class MaskedABDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob

        self._online = Proportional(support)
        self._historical = Proportional(support)
        self._dist = MixtureDistribution(
            [
                SubDistribution(d=self._online, p=left_prob),
                SubDistribution(d=self._historical, p=1 - left_prob),
            ]
        )

    def size(self):
        # define the number of elements in this buffer
        # as the total number of non-zero elements in either
        # distribution --- represented by the sum of the sumtree
        # since elements are either 1 or 0
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

        online_mask = np.full(batch_size, mode == DataMode.ONLINE)

        self._online.update(elements, ensemble_mask & online_mask)
        self._historical.update(elements, ensemble_mask & ~online_mask)


def _to_tensor(elem: object):
    if isinstance(elem, torch.Tensor | np.ndarray | list):
        return torch.Tensor(elem)
    elif elem is None:
        return torch.empty((1, 0))
    else:
        return torch.Tensor([elem])


def _get_size(experience: Transition) -> list[tuple]:
    size = []
    for elem in experience:
        if isinstance(elem, np.ndarray):
            size.append(elem.shape)
        elif isinstance(elem, torch.Tensor):
            size.append(tuple(elem.shape))
        elif isinstance(elem, int | float | bool):
            size.append((1,))
        elif isinstance(elem, list):
            size.append((len(elem),))
        else:
            raise TypeError(f"unknown type {type(elem)}")

    return size
