from typing import Literal, Protocol

import numpy as np
import torch

from corerl.configs.config import MISSING, config
from corerl.configs.group import Group

bootstrap_reduct_group = Group[
    [torch.Tensor, int],
    torch.Tensor,
]()

@config(frozen=True)
class Reduct:
    name: str = MISSING


@config(frozen=True)
class MinReduct(Reduct):
    name: Literal['min'] = 'min'

@bootstrap_reduct_group.dispatcher
def min_reduct(cfg: MinReduct, x: torch.Tensor, dim: int):
    return torch.min(x, dim=dim)[0]


@config(frozen=True)
class MaxReduct(Reduct):
    name: Literal['max'] = 'max'

@bootstrap_reduct_group.dispatcher
def max_reduct(cfg: MaxReduct, x: torch.Tensor, dim: int):
    return torch.max(x, dim=dim)[0]


@config(frozen=True)
class MeanReduct(Reduct):
    name: Literal['mean'] = 'mean'

@bootstrap_reduct_group.dispatcher
def mean_reduct(cfg: MeanReduct, x: torch.Tensor, dim: int):
    return torch.mean(x, dim=dim)


@config(frozen=True)
class MedianReduct(Reduct):
    name: Literal['median'] = 'median'

@bootstrap_reduct_group.dispatcher
def median_reduct(cfg: MedianReduct, x: torch.Tensor, dim: int):
    return torch.quantile(x, q=0.5, dim=dim)


class Statistic(Protocol):
    def __call__(self, input: torch.Tensor, dim: int) -> torch.Tensor:
        ...

@config(frozen=True)
class PercentileReduct(Reduct):
    name: Literal['percentile'] = 'percentile'
    bootstrap_batch_size: int = 10
    bootstrap_samples: int = 10
    percentile: float = 0.5

@bootstrap_reduct_group.dispatcher
def percentile_bootstrap(
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
