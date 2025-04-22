from collections.abc import Callable

import torch


def rejection_sample(
    sampler: Callable[[], torch.Tensor],
    predicate: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 100,
    fallback: Callable[[], torch.Tensor] | None = None,
):
    out = sampler()
    to_keep = predicate(out)
    n_samples = out.shape[0]

    valid = out[to_keep]

    for _ in range(max_iter - 1):
        needed = n_samples - valid.shape[0]
        if needed <= 0:
            # since we just concat all valid samples from prior round
            # it's possible we actually have too many samples now
            return valid[:n_samples]

        next_out = sampler()
        next_keep = predicate(next_out)

        valid = torch.concatenate((valid, next_out[next_keep]), dim=0)

    # check the last iteration
    needed = n_samples - valid.shape[0]
    if needed <= 0:
        return valid[:n_samples]

    # if no fallback, just return what we have
    if fallback is None:
        return valid

    fb = fallback()
    return torch.concatenate((valid, fb[:needed]), dim=0)
