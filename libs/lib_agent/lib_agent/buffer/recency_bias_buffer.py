from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple

import numpy as np
from discrete_dists.distribution import Support
from discrete_dists.mixture import MixtureDistribution, SubDistribution
from discrete_dists.proportional import Proportional
from discrete_dists.utils.SumTree import SumTree

from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.buffer.datatypes import DataMode


@dataclass
class RecencyBiasBufferConfig:
    name: Literal["recency_bias_buffer"] = "recency_bias_buffer"
    obs_period: int = 1000000
    gamma: list[float] | None = None
    effective_episodes: list[int] | None = None
    n_ensemble: int = 2
    uniform_weight: float = 0.01
    ensemble_probability: float = 0.5
    batch_size: int = 32
    max_size: int = 1_000_000
    seed: int = 0
    n_most_recent: int = 1
    id: str = ""

    def __post_init__(self):
        if self.gamma is None:
            self.gamma = [0.99] * self.n_ensemble
        if self.effective_episodes is None:
            self.effective_episodes = [100] * self.n_ensemble


class RecencyBiasBuffer[T: NamedTuple](EnsembleReplayBuffer[T]):
    def __init__(
        self,
        obs_period: int = 1000000,
        gamma: list[float] | None = None,
        effective_episodes: list[int] | None = None,
        n_ensemble: int = 2,
        uniform_weight: float = 0.01,
        ensemble_probability: float = 0.5,
        batch_size: int = 32,
        max_size: int = 1_000_000,
        seed: int = 0,
        n_most_recent: int = 1,
        id: str = "",
    ):
        if gamma is None:
            gamma = [0.99]
        if effective_episodes is None:
            effective_episodes = [100]

        super().__init__(
            n_ensemble=n_ensemble,
            ensemble_probability=ensemble_probability,
            batch_size=batch_size,
            max_size=max_size,
            seed=seed,
            n_most_recent=n_most_recent,
        )
        self._obs_period = np.timedelta64(int(obs_period), 'us')
        self._last_timestamp = None

        assert len(gamma) == n_ensemble, "Number of gamma values must match ensemble size"
        assert len(effective_episodes) == n_ensemble, "Number of effective_episodes must match ensemble size"

        self._discount_factors = [
            np.power(g, 1. / episodes)
            for g, episodes in zip(gamma, effective_episodes, strict=False)
        ]

        self._ens_dists = [
            MaskedUGDistribution(max_size, uniform_weight, ensemble_probability)
            for _ in range(n_ensemble)
        ]

        self.app_state: Any | None = None

    def _convert_timestamp(self, timestamp: datetime | np.datetime64 | int | None) -> np.datetime64 | int:
        if timestamp is None:
            return 0
        if isinstance(timestamp, datetime):
            utc_ts = timestamp.astimezone(UTC)
            naive_ts = utc_ts.replace(tzinfo=None)
            return np.datetime64(naive_ts)
        return timestamp

    def _calculate_steps(self, curr: datetime | np.datetime64 | int,
                         prev: datetime | np.datetime64 | int | None) -> float:
        if prev is None:
            return 0.0

        curr = self._convert_timestamp(curr)
        prev = self._convert_timestamp(prev)

        if isinstance(curr, np.datetime64) and isinstance(prev, np.datetime64):
            return float((curr - prev) / self._obs_period)
        if isinstance(curr, int) and isinstance(prev, int):
            return float(curr - prev)
        raise ValueError(f"Cannot mix datetime and int timestamps, curr: {curr}, prev: {prev}")

    def _calculate_timestamps(
        self,
        timestamp: datetime | np.datetime64 | int | None,
    ) -> tuple[np.datetime64 | int, float]:
        if timestamp is None:
            return 0, 0.0

        timestamp = self._convert_timestamp(timestamp)
        last_ts = self._last_timestamp if self._last_timestamp is not None else timestamp

        if isinstance(timestamp, np.datetime64):
            if isinstance(last_ts, datetime):
                last_ts = np.datetime64(last_ts)
            curr_timestamp = np.maximum(timestamp, last_ts)
        else:
            if isinstance(last_ts, np.datetime64):
                last_ts = int(last_ts)
            curr_timestamp = max(timestamp, last_ts)

        steps_since_transition = self._calculate_steps(curr_timestamp, timestamp)
        return curr_timestamp, steps_since_transition

    def add(self, transition: T, timestamp: datetime | np.datetime64 | int | None = None) -> None:
        super().add(transition)
        idx = self.size - 1

        if timestamp is None:
            return

        curr_timestamp, steps_since_transition = self._calculate_timestamps(timestamp)

        for i, (dist, discount_factor) in enumerate(zip(self._ens_dists, self._discount_factors, strict=False)):
            mask = self.ensemble_masks[i, idx]
            if self._last_timestamp is not None:
                steps_since_last_call = self._calculate_steps(curr_timestamp, self._last_timestamp)
                dist.discount_geometric(discount_factor**steps_since_last_call)
            if mask:
                weights = np.power(discount_factor, steps_since_transition)
                dist.update_uniform(np.array([idx]), np.array([True]))
                dist.update_geometric(np.array([idx]), np.array([weights]))

        self._last_timestamp = curr_timestamp

    def feed(self, transitions: Sequence[T], data_mode: DataMode) -> np.ndarray:
        idxs = np.empty(len(transitions), dtype=np.int64)
        for j, transition in enumerate(transitions):
            idxs[j] = self._storage.add(transition)

        batch_size = len(idxs)
        ensemble_masks = self._get_ensemble_masks(batch_size)

        for dist, mask in zip(self._ens_dists, ensemble_masks, strict=True):
            if mask.any():
                dist.update_uniform(idxs, mask)

        return idxs

    def _get_ensemble_masks(self, batch_size: int) -> np.ndarray:
        ensemble_masks = self.rng.random((self.n_ensemble, batch_size)) < self.ensemble_probability

        no_ensemble = ~ensemble_masks.any(axis=0)

        for idx in np.where(no_ensemble)[0]:
            random_member = self.rng.integers(0, self.n_ensemble)
            ensemble_masks[random_member, idx] = True
        return ensemble_masks

    def get_probability(self, ens_i: int, idxs: np.ndarray):
        return self._ens_dists[ens_i].probs(idxs)

    def sample(self):
        ens_idxs: list[np.ndarray] = []
        for m in range(self.n_ensemble):
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
        """
        Discount the values in the distribution by the given discount factor.
        """
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

