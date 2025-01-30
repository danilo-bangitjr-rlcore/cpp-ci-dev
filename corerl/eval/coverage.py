from dataclasses import dataclass, field
from typing import List, Protocol, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KDTree

from corerl.config import config


@dataclass
class Dataset:
    data: pd.DataFrame
    action_tags: list[str] = field(default_factory=list)
    state_tags: list[str] = field(default_factory=list)
    reward_tags: list[str] = field(default_factory=list)

    @property
    def size(self):
        return len(self.data)

    @property
    def action_dim(self):
        if not self.action_tags:
            return 0
        return len(self.action_tags)

    def subset(self, data: pd.DataFrame) -> "Dataset":
        return Dataset(data, self.action_tags, self.state_tags, self.reward_tags)


# ---------------------------- coverage functions ---------------------------- #


class CoverageProtocol(Protocol):
    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        """
        Returns an unnormalized coverage value for state, action pairs. This coverage is in [0, inf), where
        higher values indicate "less covered".
        """
        ...

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        """
        Returns an normalized coverage value for state, action pairs.
        """
        ...


@config()
class BaseCoverageConfig:
    epsilon: float = 0.01
    n_norm_samples: int = 1000


class KDECoverage:
    def __init__(self, cfg: BaseCoverageConfig):
        self.estimator = None
        self.cfg = cfg
        self.norm_const = None

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.estimator is not None
        # scipy expects df to have shape (#dim, #dimensions)
        density = self.estimator(state_action.T)  # in [0, inf)
        mapped_density = np.arctan(density) * (2 / np.pi)  # in [0, 1]
        return 1 - mapped_density  # makes 0 correspond to "covered"

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.norm_const is not None
        return self.unnorm_cov(state_action) / self.norm_const

    def fit(self, dataset: Dataset) -> None:
        # scipy expects df to have shape (#dim, #dimensions)
        self.estimator = stats.gaussian_kde(dataset.data.transpose())
        self.norm_const = get_norm_const(self, dataset, self.cfg.epsilon, self.cfg.n_norm_samples)


@config()
class NeighboursCoverageConfig(BaseCoverageConfig):
    n_neighbours: int = 1
    metric: str = "l2"


class NeighboursCoverage:
    def __init__(self, cfg: NeighboursCoverageConfig):
        self.tree = None
        self.cfg = cfg
        self.norm_const = None
        self.mapping = lambda x: x

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.tree is not None
        dist, _ = self.tree.query(state_action, k=self.cfg.n_neighbours)
        return np.mean(dist, axis=1)

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.norm_const is not None
        return self.unnorm_cov(state_action) / self.norm_const

    def fit(self, dataset: Dataset) -> None:
        self.tree = KDTree(dataset.data.to_numpy(), metric=self.cfg.metric)
        self.norm_const = get_norm_const(self, dataset, self.cfg.epsilon, self.cfg.n_norm_samples)


# --------------------------------- utilities -------------------------------- #


def sample_epsilon_ball(center: np.ndarray, epsilon: float, n_samples: int = 1) -> np.ndarray:
    dimension = len(center)
    directions = np.random.normal(0, 1, (n_samples, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = center + epsilon * directions
    return points


def get_norm_const(coverage_fn: CoverageProtocol, dataset: Dataset, epsilon: float = 0.01, n_samples: int = 1000):
    data = dataset.data
    cumulative_coverage = 0
    for _, row in data.iterrows():
        row_np = row.to_numpy()
        samples = sample_epsilon_ball(row_np, epsilon, n_samples)
        coverages = coverage_fn.unnorm_cov(samples)
        cumulative_coverage += np.sum(coverages)
    return cumulative_coverage / (n_samples * dataset.size)


# --------------------------------- samplers --------------------------------- #


@dataclass
class BaseSamplerConfig:
    n_state_samples: int = 1
    action_low: np.ndarray | float = 0.0
    action_high: np.ndarray | float = 1.0


class UniformDatasetSampler:
    def __init__(self, cfg: BaseSamplerConfig):
        self.cfg = cfg

    def eval(self, dataset: Dataset, coverage_fn: CoverageProtocol) -> float:
        sampled_data = dataset.data.sample(n=self.cfg.n_state_samples)
        coverages = coverage_fn.cov(sampled_data.to_numpy())
        return coverages.mean()


@dataclass
class ActionSamplerConfig(BaseSamplerConfig):
    n_action_samples: int = 1


class UniformActionSampler:
    def __init__(self, cfg: ActionSamplerConfig):
        self.cfg = cfg

    def eval(self, dataset: Dataset, coverage_fn: CoverageProtocol) -> float:
        sampled_data = dataset.data.sample(n=self.cfg.n_state_samples)
        repeated_samples = pd.concat([sampled_data] * self.cfg.n_state_samples, ignore_index=True)
        sampled_actions = np.random.uniform(
            self.cfg.action_low,
            self.cfg.action_high,
            (self.cfg.n_state_samples * self.cfg.n_state_samples, dataset.action_dim),
        )
        repeated_samples[dataset.action_tags] = sampled_actions
        coverages = coverage_fn.cov(repeated_samples.to_numpy())
        return float(np.mean(coverages))
