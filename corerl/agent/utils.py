from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

from corerl.component.network.networks import EnsembleNetworkReturn
from corerl.utils.device import device

if TYPE_CHECKING:
    from corerl.component.policy_manager import ActionReturn


def grab_top_n(
    values: torch.Tensor,
    keys: list[torch.Tensor],
    n: int,
) -> list[torch.Tensor]:
    sanitize_grab_top_n(values, keys)
    if n == 1:
        sorted_inds = torch.argmax(values, dim=1, keepdim=True)
    else:
        sorted_inds = torch.argsort(values, dim=1, descending=True)

    top_n_indices = sorted_inds[:, :n]
    top_n_indices = top_n_indices.unsqueeze(-1)

    selected_keys_list = []
    for k in keys:
        k_dim = k.size(2)
        top_n_indices_k = top_n_indices.repeat_interleave(k_dim, -1) # batch x top_n x (state_dim or action_dim)
        selected_keys = torch.gather(k, dim=1, index=top_n_indices_k)
        selected_keys_list.append(selected_keys)

    return selected_keys_list

def sanitize_grab_top_n(
        values: torch.Tensor,
    keys: list[torch.Tensor],
):
    """
    In policy update:
        values  <- action values with dim: batch x samples
        keys[0] <- states with        dim: batch x samples x state dim
        keys[1] <- actions with       dim: batch x samples x action dim
    """
    assert len(keys) > 0
    for k in keys:
        # ensure all keys have the same shape up to first two axes
        assert k.dim() == 3
        assert k.size(0) == keys[0].size(0)
        assert k.size(1) == keys[0].size(1)

    assert values.dim() == 2

def grab_percentile(
    values: torch.Tensor,
    keys: list[torch.Tensor],
    percentile: float,
) -> list[torch.Tensor]:
    n_samples = values.size(1)
    top_n = ceil(percentile * n_samples)
    return grab_top_n(values, keys, top_n)

def get_percentile_threshold(
    q_vals: torch.Tensor,
    percentile: float,
):
    assert q_vals.dim() == 2
    n_samples = q_vals.size(1)
    top_n = ceil(percentile*n_samples)
    top_n_values, _ = torch.topk(q_vals, k=top_n, dim=1)
    return top_n_values.min(dim=1).values

class SampledQReturn(NamedTuple):
    q_values: torch.Tensor
    states: torch.Tensor
    direct_actions: torch.Tensor
    policy_actions: torch.Tensor

class Sampler(Protocol):
    def __call__(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn: ...

class ValueEstimator(Protocol):
    def get_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
    ) -> EnsembleNetworkReturn: ...


def get_sampled_qs(
    states: torch.Tensor,
    action_lo: torch.Tensor,
    action_hi: torch.Tensor,
    n_samples: int,
    sampler: Sampler,
    critic: ValueEstimator
) -> SampledQReturn:
    batch_size = states.size(0)

    ar = sampler(
        n_samples,
        states,
        action_lo,
        action_hi,
    )

    repeated_states = states.repeat_interleave(n_samples, dim=0)
    actions = ar.direct_actions.reshape(batch_size * n_samples, -1)
    q_values = critic.get_values([repeated_states], [actions]).reduced_value
    q_values = q_values.reshape(batch_size, n_samples)

    states = repeated_states.reshape(batch_size, n_samples, -1)

    return SampledQReturn(q_values, states, ar.direct_actions, ar.policy_actions)

def mix_uniform_actions(policy_actions: torch.Tensor, uniform_weight: float) -> torch.Tensor:
    batch_size = policy_actions.size(0)
    action_dim = policy_actions.size(1)
    num_rows_to_sample = ceil(batch_size * uniform_weight)
    indices = torch.randperm(batch_size)[:num_rows_to_sample]
    rand_actions = torch.rand(num_rows_to_sample, action_dim, device=device.device)
    rand_actions = torch.clip(rand_actions, 0, 1)
    policy_actions[indices, :] = rand_actions
    return policy_actions
