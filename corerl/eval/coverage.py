from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KDTree
from tqdm import tqdm

from corerl.component.network.utils import tensor, to_np
from corerl.config import config
from corerl.data_pipeline.imputers.auto_encoder import CircularBuffer


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
        self.tree: KDTree | None = None
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


@config()
class AECoverageConfig(BaseCoverageConfig):
    buffer_size: int = 50_000
    batch_size: int = 256
    stepsize: float = 1e-3
    weight_decay: float = 0.01
    err_tolerance: float = 1e-3
    max_update_steps: int = 10000
    sizes: list[int] | None = None
    ball_in_batch = True


def weights_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class AECoverage:
    def __init__(self, cfg: AECoverageConfig):
        self.cfg = cfg
        self.norm_const = None
        self._model = None
        self._optimizer = None
        self._buffer = None

    def unnorm_cov(self, state_action: np.ndarray) -> np.ndarray:
        state_action_tensor = tensor(state_action)
        return to_np(self.loss(state_action_tensor))

    def cov(self, state_action: np.ndarray) -> np.ndarray:
        assert self.norm_const is not None
        return self.unnorm_cov(state_action) / self.norm_const

    def _set_up(self, dataset: Dataset):
        self._input_dim = len(dataset.data.columns)

        if self.cfg.sizes is None:
            sizes = [
                int(self._input_dim),
                int(np.ceil(0.75 * self._input_dim)),
                int(np.ceil(0.5 * self._input_dim)),
                int(np.ceil(0.75 * self._input_dim)),
                int(self._input_dim),
            ]
        else:
            sizes = [int(self._input_dim)] + self.cfg.sizes + [int(self._input_dim)]

        parts: list[nn.Module] = []

        for i in range(1, len(sizes)):
            parts.append(nn.Linear(sizes[i - 1], sizes[i]))
            parts.append(nn.LeakyReLU(negative_slope=0.1))

        self._model = nn.Sequential(*parts)
        self._model.apply(weights_init)

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.cfg.stepsize,
            weight_decay=self.cfg.weight_decay,
        )
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, "min", factor=0.5, patience=100, min_lr=1e-6, threshold=1e-4
        )

        self._buffer = CircularBuffer(self.cfg.buffer_size)

    def fit(self, dataset: Dataset) -> None:
        if self._model is None:
            self._set_up(dataset)

        assert self._buffer is not None
        assert self._optimizer is not None

        data_tensor = tensor(dataset.data.to_numpy())
        for row in data_tensor:
            self._buffer.add(row)

        steps = 0
        loss = torch.inf
        pbar = tqdm(total=self.cfg.max_update_steps, desc="Training")
        while loss > self.cfg.err_tolerance and steps < self.cfg.max_update_steps:
            steps += 1
            self._optimizer.zero_grad()

            if self.cfg.ball_in_batch:  # add some epsilon perturbations to the batch
                batch_1 = self._buffer.sample(self.cfg.batch_size // 2)

                batch_2 = tensor(sample_epsilon_ball(to_np(batch_1), self.cfg.epsilon, 1))
                batch = torch.concat([batch_1, batch_2], dim=0)

            else:
                batch = self._buffer.sample(self.cfg.batch_size)

            loss = self.loss(batch, with_grad=True).mean()  # extra sum for summing loss across elements of batch
            loss.backward()

            self._optimizer.step()
            self._scheduler.step(loss.item())

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{self._optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        self.norm_const = get_norm_const(self, dataset, self.cfg.epsilon, self.cfg.n_norm_samples)

    def loss(self, batch: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        assert self._model is not None
        context = torch.no_grad() if not with_grad else nullcontext()
        with context:
            out = self._model(batch)
            loss_by_sample = torch.square(batch - out).mean(dim=1)

        return loss_by_sample


# --------------------------------- utilities -------------------------------- #


def sample_epsilon_ball(
    center: np.ndarray,
    epsilon: float,
    n_samples: int = 1,
) -> np.ndarray:
    """
    Returns n_samples from an epsilon ball around center.
    """
    if len(center.shape) == 1:
        center = np.expand_dims(center, axis=0)

    dimension = center.shape[1]
    n_center_samples = center.shape[0]
    directions = np.random.normal(0, 1, (n_samples * n_center_samples, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    repeated_center = np.repeat(center, repeats=n_samples, axis=0)
    points = repeated_center + epsilon * directions
    return points


def get_norm_const(
    coverage_fn: CoverageProtocol,
    dataset: Dataset,
    epsilon: float = 0.01,
    n_samples: int = 1000,
) -> float:
    data = dataset.data
    samples = sample_epsilon_ball(data.to_numpy(), epsilon, n_samples)
    coverages = coverage_fn.unnorm_cov(samples)
    return float(np.mean(coverages))


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
