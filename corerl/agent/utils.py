from math import ceil
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch

from corerl.utils.device import device

if TYPE_CHECKING:
    from corerl.component.critic.ensemble_critic import EnsembleCritic
    from corerl.component.policy_manager import ActionReturn

def grab_percentile(
        values: torch.Tensor,
        keys: list[torch.Tensor],
        percentile: float,
    ) -> list[torch.Tensor]:
    assert len(keys) > 0
    for k in keys:
        assert k.dim() == 3
        assert k.size(0) == keys[0].size(0)
        assert k.size(1) == keys[0].size(1)

    assert values.dim() == 2

    n_samples = values.size(1)
    top_n = ceil(percentile * n_samples)

    sorted_inds = torch.argsort(values, dim=1, descending=True)
    top_n_indices = sorted_inds[:, :top_n]
    top_n_indices = top_n_indices.unsqueeze(-1)

    selected_keys_list = []
    for k in keys:
        k_dim = k.size(2)
        top_n_indices_k = top_n_indices.repeat_interleave(k_dim, -1)
        selected_keys = torch.gather(k, dim=1, index=top_n_indices_k)
        selected_keys_list.append(selected_keys)

    return selected_keys_list

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
    q_values : torch.Tensor
    states : torch.Tensor
    direct_actions : torch.Tensor
    policy_actions : torch.Tensor

Sampler = Callable[[torch.Tensor, torch.Tensor], "ActionReturn"]

def get_sampled_qs(
    states: torch.Tensor,
    prev_actions: torch.Tensor,
    n_samples: int,
    sampler: Sampler ,
    critic: "EnsembleCritic"
    ) -> SampledQReturn:

    batch_size = states.size(0)

    repeated_states = states.repeat_interleave(n_samples, dim=0)
    repeated_prev_a = prev_actions.repeat_interleave(n_samples, dim=0)
    ar = sampler(
        repeated_states,
        repeated_prev_a,
    )
    q_values = critic.get_values([repeated_states], [ar.direct_actions]).reduced_value
    q_values = q_values.reshape(batch_size, n_samples)

    states = repeated_states.reshape(batch_size, n_samples, -1)
    direct_actions = ar.direct_actions.reshape(batch_size, n_samples, -1)
    policy_actions = ar.policy_actions.reshape(batch_size, n_samples, -1)

    return SampledQReturn(q_values, states, direct_actions, policy_actions)

def mix_uniform_actions(policy_actions: torch.Tensor, uniform_weight: float) -> torch.Tensor:
    batch_size = policy_actions.size(0)
    action_dim = policy_actions.size(1)
    num_rows_to_sample = ceil(batch_size * uniform_weight)
    indices = torch.randperm(batch_size)[:num_rows_to_sample]
    rand_actions = torch.rand(num_rows_to_sample, action_dim, device=device.device)
    rand_actions = torch.clip(rand_actions, 0, 1)
    policy_actions[indices, :] = rand_actions
    return policy_actions

def mix_uniform_actions_evenly_dispersed(
        policy_actions: torch.Tensor,
        uniform_weight: float,
    ) -> torch.Tensor:
    batch_size = policy_actions.size(0)
    action_dim = policy_actions.size(1)

    num_rows_to_sample = ceil(batch_size * uniform_weight)

    if num_rows_to_sample == 0:
        return policy_actions

    if num_rows_to_sample >= batch_size:
        rand_actions = torch.rand(batch_size, action_dim, device=policy_actions.device)
        rand_actions = torch.clip(rand_actions, 0, 1)
        return rand_actions

    # Generate evenly spaced indices
    stride = batch_size / num_rows_to_sample
    indices = torch.tensor([int(i * stride) for i in range(num_rows_to_sample)],
                          device=policy_actions.device)

    rand_actions = torch.rand(num_rows_to_sample, action_dim, device=policy_actions.device)
    rand_actions = torch.clip(rand_actions, 0, 1)
    policy_actions[indices, :] = rand_actions

    return policy_actions
