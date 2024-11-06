import torch
import numpy as np

from dataclasses import dataclass
from corerl.utils.hydra import Group

ensemble_bootstrap_reduct_group = Group(
    'agent/critic/critic_network/bootstrap_reduct',
    return_type=torch.Tensor,
)

ensemble_policy_reduct_group = Group(
    'agent/critic/critic_network/policy_reduct',
    return_type=torch.Tensor,
)


@dataclass
class MinReduct:
    name: str = 'min'

@ensemble_bootstrap_reduct_group.dispatcher
@ensemble_policy_reduct_group.dispatcher
def _min_reduct(cfg: MinReduct, x: torch.Tensor, dim: int):
    return torch.min(x, dim=dim)[0]


@dataclass
class MaxReduct:
    name: str = 'max'

@ensemble_bootstrap_reduct_group.dispatcher
@ensemble_policy_reduct_group.dispatcher
def _max_reduct(cfg: MaxReduct, x: torch.Tensor, dim: int):
    return torch.max(x, dim=dim)[0]


@dataclass
class MeanReduct:
    name: str = 'mean'

@ensemble_bootstrap_reduct_group.dispatcher
@ensemble_policy_reduct_group.dispatcher
def _mean_reduct(cfg: MeanReduct, x: torch.Tensor, dim: int):
    return torch.mean(x, dim=dim)


@dataclass
class MedianReduct:
    name: str = 'median'

@ensemble_bootstrap_reduct_group.dispatcher
@ensemble_policy_reduct_group.dispatcher
def _median_reduct(cfg: MedianReduct, x: torch.Tensor, dim: int):
    return torch.quantile(x, q=0.5, dim=dim)


@dataclass
class PercentileReduct:
    name: str = 'percentile'
    bootstrap_batch_size: int = 10
    bootstrap_samples: int = 10
    percentile: float = 0.5

@ensemble_bootstrap_reduct_group.dispatcher
@ensemble_policy_reduct_group.dispatcher
def _percentile_bootstrap(
    cfg: PercentileReduct,
    x: torch.Tensor,
    dim: int,
    statistic=torch.mean,
):
    size = (
        *x.shape[:dim],
        cfg.bootstrap_batch_size * cfg.bootstrap_samples,
        *x.shape[dim + 1:],
    )

    # Randomly sampling integers from numpy is faster than from torch
    ind = np.random.randint(0, x.shape[dim], size)
    samples = torch.gather(x, dim, torch.from_numpy(ind))

    size = (
        *x.shape[:dim],
        cfg.bootstrap_batch_size,
        cfg.bootstrap_samples,
        *x.shape[dim + 1:],
    )
    samples = samples.reshape(size)
    bootstr_stat = statistic(samples, dim=(len(x.shape[:dim])))

    return torch.quantile(bootstr_stat, cfg.percentile, dim=dim)
