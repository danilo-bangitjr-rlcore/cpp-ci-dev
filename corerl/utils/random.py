from collections.abc import Callable
from typing import NamedTuple

import torch


def rejection_sample(
    sampler: Callable[[int], torch.Tensor],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int,
    max_iter: int = 100,
    fallback: Callable[[int], torch.Tensor] | None = None,
):
    out = sampler(n_samples)
    to_keep = predicate(out)
    assert to_keep.shape == (n_samples,)

    valid = out[to_keep]

    for _ in range(max_iter - 1):
        needed = n_samples - valid.shape[0]
        if needed <= 0:
            # since we just concat all valid samples from prior round
            # it's possible we actually have too many samples now
            return valid[:n_samples]

        next_out = sampler(n_samples)
        next_keep = predicate(next_out)

        valid = torch.concatenate((valid, next_out[next_keep]), dim=0)

    # check the last iteration
    needed = n_samples - valid.shape[0]
    if needed <= 0:
        return valid[:n_samples]

    # if no fallback, just return what we have
    if fallback is None:
        return valid

    fb = fallback(needed)
    return torch.concatenate((valid, fb), dim=0)


class DistributionStatistics(NamedTuple):
    mean: torch.Tensor
    stddev: torch.Tensor


def get_dist_stats(dist: torch.distributions.Distribution, n_samples: int = 100) -> DistributionStatistics:
    samples = dist.sample((n_samples,))
    mean = samples.mean(dim=0)
    stddev = samples.std(dim=0)
    return DistributionStatistics(
        mean=mean,
        stddev=stddev,
    )
