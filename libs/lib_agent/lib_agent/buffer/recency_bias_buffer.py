from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypeVar

import numpy as np
from discrete_dists.distribution import Support
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.utils.SumTree import SumTree

from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.buffer.datatypes import DataMode

# NOTE: the python 3.12+ syntax for generic types is not compatible with pickle
T = TypeVar('T', bound=NamedTuple)


@dataclass
class RecencyBiasBufferConfig:
    obs_period: int
    name: Literal["recency_bias_buffer"] = "recency_bias_buffer"
    gamma: Sequence[float] = field(default_factory=lambda: [0.99])
    effective_episodes: Sequence[int] = field(default_factory=lambda: [100])
    ensemble: int = 2
    uniform_weight: float = 0.01
    ensemble_probability: float = 0.5
    batch_size: int = 32
    max_size: int = 1_000_000
    seed: int = 0
    n_most_recent: int = 1
    id: str = ""


class RecencyBiasBuffer(EnsembleReplayBuffer[T]):
    def __init__(
        self,
        obs_period: int,
        gamma: Sequence[float] = [0.99],
        effective_episodes: Sequence[int] = [100],
        ensemble: int = 2,
        uniform_weight: float = 0.01,
        ensemble_probability: float = 0.5,
        batch_size: int = 32,
        max_size: int = 1_000_000,
        seed: int = 0,
        n_most_recent: int = 1,
        id: str = "",
    ):
        super().__init__(
            ensemble=ensemble,
            ensemble_probability=ensemble_probability,
            batch_size=batch_size,
            max_size=max_size,
            seed=seed,
            n_most_recent=n_most_recent,
        )
        self._obs_period = obs_period
        self._last_timestamp: int | None = None

        assert len(gamma) == ensemble, "Number of gamma values must match ensemble size"
        assert len(effective_episodes) == ensemble, "Number of effective_episodes must match ensemble size"

        self._discount_factors = [
            np.power(g, 1. / episodes)
            for g, episodes in zip(gamma, effective_episodes, strict=False)
        ]

        self._ens_dists = [
            MaskedUGDistribution(max_size, uniform_weight, ensemble_probability)
            for _ in range(ensemble)
        ]

    def _calculate_steps(self, curr: int, prev: int | None) -> float:
        if prev is None:
            return 0.0
        return float(curr - prev) / self._obs_period

    def _calculate_timestamps(self, timestamp: int | None) -> tuple[int, float]:
        if timestamp is None:
            return 0, 0.0

        last_ts = self._last_timestamp if self._last_timestamp is not None else timestamp
        curr_timestamp = max(timestamp, last_ts)
        steps_since_transition = self._calculate_steps(curr_timestamp, timestamp)
        return curr_timestamp, steps_since_transition

    def _update_distributions(self, idx: int, timestamp: int, mask: np.ndarray) -> None:
        curr_timestamp, steps_since_transition = self._calculate_timestamps(timestamp)

        for i, (dist, discount_factor) in enumerate(zip(self._ens_dists, self._discount_factors, strict=False)):
            if self._last_timestamp is not None:
                steps_since_last_call = self._calculate_steps(curr_timestamp, self._last_timestamp)
                dist.discount_geometric(discount_factor**steps_since_last_call)
            if mask[i]:
                weights = np.power(discount_factor, steps_since_transition)
                dist.update_uniform(np.array([idx]), np.array([True]))
                dist.update_geometric(np.array([idx]), np.array([weights]))

        self._last_timestamp = curr_timestamp

    def add(self, transition: T) -> None:
        super().add(transition)
        idx = self.size - 1

        timestamp = getattr(transition, 'timestamp', None)
        assert timestamp is not None, "RecencyBiasBuffer requires transitions with timestamps"

        self._update_distributions(idx, timestamp, self.ensemble_masks[:, idx])

    def feed(self, transitions: Sequence[T], data_mode: DataMode) -> np.ndarray:
        idxs = np.empty(len(transitions), dtype=np.int64)
        batch_size = len(transitions)
        ensemble_masks = self._get_ensemble_masks(batch_size)

        for j, transition in enumerate(transitions):
            idx = self._storage.add(transition)
            idxs[j] = idx

            timestamp = getattr(transition, 'timestamp', None)
            assert timestamp is not None, "RecencyBiasBuffer requires transitions with timestamps"

            self._update_distributions(idx, timestamp, ensemble_masks[:, j])

        return idxs

    def _get_ensemble_masks(self, batch_size: int) -> np.ndarray:
        ensemble_masks = self.rng.random((self.ensemble, batch_size)) < self.ensemble_probability

        no_ensemble = ~ensemble_masks.any(axis=0)

        for idx in np.where(no_ensemble)[0]:
            random_member = self.rng.integers(0, self.ensemble)
            ensemble_masks[random_member, idx] = True
        return ensemble_masks

    def get_probability(self, ens_i: int, idxs: np.ndarray):
        return self._ens_dists[ens_i].probs(idxs)

    def sample(self):
        ens_idxs: list[np.ndarray] = []
        for m in range(self.ensemble):
            valid_indices = np.nonzero(self.ensemble_masks[m, :self._storage.size()])[0]
            if len(valid_indices) == 0:
                continue

            sampled_indices = self._ens_dists[m].sample(self.rng, self.batch_size)
            sampled_indices = sampled_indices % len(valid_indices)
            ens_idxs.append(valid_indices[sampled_indices])

        return self._storage.get_ensemble_batch(ens_idxs)

    def get_batch(self, idxs: np.ndarray):
        return self._storage.get_batch(idxs)

    @property
    def ensemble_sizes(self) -> list[int]:
        return [d.size() for d in self._ens_dists]

    @property
    def is_sampleable(self) -> bool:
        return min(self.ensemble_sizes) > 0


class Geometric(Proportional):
    def __init__(self, support: Support | int):
        if isinstance(support, int):
            support = (0, support)

        self._support = support
        rang = support[1] - support[0]
        self.tree = SumTree(rang)

    def discount(self, discount_factor: float):
        old_values = self.tree.get_values(np.arange(self._support[0], self._support[1]))
        new_values = old_values * discount_factor
        self.tree.update(np.arange(self._support[0], self._support[1]), new_values)


class MaskedUGDistribution:
    def __init__(self, support: int, left_prob: float, mask_prob: float):
        self._mask_prob = mask_prob
        self._support = (0, support)

        self._uniform = Proportional(self._support)
        self._geometric = Geometric(self._support)
        self._dist = MixtureDistribution(
            [
                SubDistribution(d=self._uniform, p=left_prob),
                SubDistribution(d=self._geometric, p=1-left_prob),
            ],
        )

    def size(self):
        return int(self._uniform.tree.total())

    def probs(self, elements: np.ndarray) -> np.ndarray:
        return self._dist.probs(elements)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._dist.sample(rng, n)

    def update_uniform(
        self,
        elements: np.ndarray,
        ensemble_mask: np.ndarray,
    ):
        self._uniform.update(elements, ensemble_mask)

    def update_geometric(
        self,
        elements: np.ndarray,
        initial_prob: np.ndarray,
    ):
        self._geometric.update(elements, initial_prob)

    def discount_geometric(
        self,
        discount: float,
    ):
        self._geometric.discount(discount)


def create_recency_bias_buffer_from_config(cfg: RecencyBiasBufferConfig) -> RecencyBiasBuffer:
    gamma = cfg.gamma if cfg.gamma is not None else [0.99] * cfg.ensemble
    effective_episodes = cfg.effective_episodes if cfg.effective_episodes is not None else [100] * cfg.ensemble

    return RecencyBiasBuffer(
        obs_period=cfg.obs_period,
        gamma=gamma,
        effective_episodes=effective_episodes,
        ensemble=cfg.ensemble,
        uniform_weight=cfg.uniform_weight,
        ensemble_probability=cfg.ensemble_probability,
        batch_size=cfg.batch_size,
        max_size=cfg.max_size,
        seed=cfg.seed,
        n_most_recent=cfg.n_most_recent,
        id=cfg.id,
    )
