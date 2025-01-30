from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KDTree


@dataclass
class Dataset:
    data : pd.DataFrame
    action_tags : list[str] | None = None
    state_tags  : list[str] | None = None
    reward_tags : list[str] | None = None

    @property
    def size(self):
        return len(self.data)

    @property
    def action_dim(self):
        if self.action_tags is None:
            return 0
        return len(self.action_tags)

    def subset(self, data: pd.DataFrame) -> "Dataset":
        return Dataset(
            data,
            self.action_tags,
            self.state_tags,
            self.reward_tags,
        )


# ---------------------------- coverage functions ---------------------------- #


class CoverageProtocol(Protocol):
    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        ...

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        ...


class KDECoverage():
    def __init__(self, epsilon: float, n_norm_samples: int):
        self.estimator = None
        self.epsilon = epsilon
        self.n_norm_samples = n_norm_samples
        self.norm_const = None

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.estimator is not None
        # scipy expects df to have shape (#dim, #dimensions)
        density = self.estimator(state_action.T)
        mapped_density = np.arctan(density) *(2/np.pi)
        return 1-mapped_density

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.norm_const is not None
        return (self.unnorm_cov(state_action) / self.norm_const)

    def fit(self, dataset: Dataset) -> None:
        # scipy expects df to have shape (#dim, #dimensions)
        self.estimator = stats.gaussian_kde(dataset.data.transpose())
        self.norm_const = get_norm_const(self, dataset, self.epsilon, self.n_norm_samples)


class NeighboursCoverage():
    def __init__(self, epsilon: float, n_norm_samples: int):
        self.estimator = None
        self.n_neighbours = 1
        self.epsilon = epsilon
        self.n_norm_samples = n_norm_samples
        self.norm_const = None
        self.mapping = lambda x: x
        self.metric = "l2"

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        dist, _ = self.tree.query(state_action, k=self.n_neighbours)
        return np.mean(dist, axis=1)

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.norm_const is not None
        return (self.unnorm_cov(state_action) / self.norm_const)

    def fit(self, dataset: Dataset) -> None:
        self.tree = KDTree(dataset.data.to_numpy(), metric=self.metric)
        self.norm_const = get_norm_const(self, dataset, self.epsilon, self.n_norm_samples)



# --------------------------------- utilities -------------------------------- #


def sample_epsilon_ball(
        center: np.ndarray,
        epsilon: float,
        n_samples:int=1
    ) -> np.ndarray:
    dimension = len(center)
    directions = np.random.normal(0, 1, (n_samples, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) # directions with unit length.
    points = center + epsilon * directions
    return points


def get_norm_const(
        coverage_fn : CoverageProtocol,
        dataset: Dataset,
        epsilon:float=0.01,
        n_samples:int=1000,
    ):
    data = dataset.data
    cumulative_coverage = 0
    for _, row in data.iterrows():
        row_np = row.to_numpy()
        samples = sample_epsilon_ball(row_np, epsilon, n_samples)
        coverages =  coverage_fn.unnorm_cov(samples)
        cumulative_coverage += np.sum(coverages)

    return cumulative_coverage / (n_samples*dataset.size)


# --------------------------------- samplers --------------------------------- #


class UniformDatasetSampler():
    def eval(
            self,
            dataset: Dataset,
            coverage_fn: CoverageProtocol,
            n_samples: int = 1
        ) -> float:
        sampled_data = dataset.data.sample(n=n_samples)
        coverages = coverage_fn.cov(sampled_data.to_numpy())
        return coverages.mean()


class UniformActionSampler():
    """
    Samples states from dataset, actions from bounds UAR
    """
    def __init__(self, action_low: np.ndarray | float = 0., action_high: np.ndarray | float = 1.):
        self.action_low = action_low
        self.action_high = action_high

    def eval(self,
            dataset: Dataset,
            coverage_fn: CoverageProtocol,
            n_state_samples: int = 1,
            n_action_samples: int = 1,
        ) -> float:

        sampled_data = dataset.data.sample(n=n_state_samples)
        repeated_samples = pd.concat([sampled_data] * n_action_samples, ignore_index=True)
        sampled_actions =  np.random.uniform(
            self.action_low,
            self.action_high,
            (n_state_samples*n_action_samples, dataset.action_dim))

        repeated_samples[dataset.action_tags] = sampled_actions
        coverages = coverage_fn.cov(repeated_samples.to_numpy())
        return float(np.mean(coverages))



