from typing import Literal, Protocol
import torch
import numpy as np

from corerl.configs.config import config
from corerl.configs.group import Group

bootstrap_reduct_group = Group[
    [torch.Tensor, int],
    torch.Tensor,
]()


@config(frozen=True)
class MinReduct:
    name: Literal['min'] = 'min'

@bootstrap_reduct_group.dispatcher
def _min_reduct(cfg: MinReduct, x: torch.Tensor, dim: int):
    return torch.min(x, dim=dim)[0]


@config(frozen=True)
class MaxReduct:
    name: Literal['max'] = 'max'

@bootstrap_reduct_group.dispatcher
def _max_reduct(cfg: MaxReduct, x: torch.Tensor, dim: int):
    return torch.max(x, dim=dim)[0]


@config(frozen=True)
class MeanReduct:
    name: Literal['mean'] = 'mean'

@bootstrap_reduct_group.dispatcher
def _mean_reduct(cfg: MeanReduct, x: torch.Tensor, dim: int):
    return torch.mean(x, dim=dim)


@config(frozen=True)
class MedianReduct:
    name: Literal['median'] = 'median'

@bootstrap_reduct_group.dispatcher
def _median_reduct(cfg: MedianReduct, x: torch.Tensor, dim: int):
    return torch.quantile(x, q=0.5, dim=dim)


class Statistic(Protocol):
    def __call__(self, input: torch.Tensor, dim: int) -> torch.Tensor:
        ...

@config(frozen=True)
class PercentileReduct:
    name: Literal['percentile'] = 'percentile'
    bootstrap_batch_size: int = 10
    bootstrap_samples: int = 10
    percentile: float = 0.5

@bootstrap_reduct_group.dispatcher
def _percentile_bootstrap(
    cfg: PercentileReduct,
    x: torch.Tensor,
    dim: int,
    statistic: Statistic = torch.mean,
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
