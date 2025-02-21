from math import floor

import torch
from jaxtyping import Float

from corerl.agent.base import BaseAC
from corerl.agent.greedy_ac import EPSILON, logger
from corerl.component.actor.base_actor import BaseActor
from corerl.utils.device import device


def get_q_for_sample(
        agent: BaseAC,
        states: Float[torch.Tensor, "batch_size n_samples state_dim"],
        direct_actions: Float[torch.Tensor, "batch_size n_samples action_dim"],
    ) ->  Float[torch.Tensor, "batch_size n_samples"]:

    assert direct_actions.dim() == states.dim()
    assert states.size(0) == direct_actions.size(0)
    assert states.size(1) == direct_actions.size(1)

    BATCH_SIZE = direct_actions.size(0)
    N_SAMPLES = direct_actions.size(1)

    states_2d = states.reshape(BATCH_SIZE * N_SAMPLES, -1)
    direct_actions_2d = direct_actions.reshape(BATCH_SIZE * N_SAMPLES, -1)

    q_values_1d = agent.q_critic.get_q(
        [states_2d],
        [direct_actions_2d],
        with_grad=False,
        bootstrap_reduct=False,
    )
    q_values_2d = q_values_1d.reshape(BATCH_SIZE, N_SAMPLES)

    return q_values_2d


def unsqueeze_repeat(tensor: torch.Tensor, dim: int, repeats: int) -> torch.Tensor:
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.repeat_interleave(repeats, dim=dim)
    return tensor


def sample_actions(
    state_batch: Float[torch.Tensor, "batch_size state_dim"],
    n_samples: int,
    action_dim: int,
    policy: BaseActor | None = None,
    uniform_weight: float = 0.0,
) -> tuple[
    Float[torch.Tensor, "batch_size num_samples action_dim"], Float[torch.Tensor, "batch_size num_samples state_dim"]
]:
    """
    For each state in the state_batch, sample n actions according to policy.

    Returns a tensor with dimensions (batch_size, num_samples, action_dim)
    """
    batch_size = state_batch.shape[0]

    policy_weight = 1 - uniform_weight
    n_samples_policy = floor(policy_weight * n_samples)  # number of samples from the policy
    n_samples_uniform = n_samples - n_samples_policy

    SAMPLE_DIM = 1

    if policy_weight > 0:
        assert policy is not None
        # sample n_samples_policy actions from the policy
        repeated_states: Float[torch.Tensor, "batch_size n_samples_policy state_dim"]
        repeated_states = unsqueeze_repeat(state_batch, SAMPLE_DIM, n_samples_policy)
        proposed_actions: Float[torch.Tensor, "batch_size n_samples_policy action_dim"]
        proposed_actions, _ = policy.get_action(repeated_states, with_grad=False)

    else:
        proposed_actions = torch.empty(batch_size, 0, action_dim)

    # sample remaining n_samples_uniform actions uniformly
    uniform_sample_actions = torch.rand(batch_size, n_samples_uniform, action_dim)
    uniform_sample_actions = torch.clip(uniform_sample_actions, EPSILON, 1)

    sample_actions = torch.cat([proposed_actions, uniform_sample_actions], dim=SAMPLE_DIM)

    repeated_states: Float[torch.Tensor, "batch_size n_samples state_dim"]
    repeated_states = unsqueeze_repeat(state_batch, SAMPLE_DIM, n_samples)

    logger.debug(f"{proposed_actions.shape=}")
    logger.debug(f"{uniform_sample_actions.shape=}")

    sample_actions.to(device.device)
    repeated_states.to(device.device)

    return sample_actions, repeated_states


def get_percentile_threshold(
        q_vals: Float[torch.Tensor, "batch_size n_samples"],
        percentile: float,
    ):
    assert q_vals.dim() == 2
    n_samples = q_vals.size(1)
    top_n = floor(percentile*n_samples)
    top_n_values, _ = torch.topk(q_vals, k=top_n, dim=1)
    return torch.mean(top_n_values, dim=1)


def get_percentile_inds(
        values: torch.Tensor,
        keys: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
    assert keys.dim() == 3
    assert values.dim() == 2
    assert values.size(0) == keys.size(0)
    assert values.size(1) == keys.size(1)
    key_dim = keys.size(2)

    n_samples = values.size(1)
    top_n = floor(percentile * n_samples)

    sorted_inds = torch.argsort(values, dim=1, descending=True)
    top_n_indices = sorted_inds[:, :top_n]
    top_n_indices = top_n_indices.unsqueeze(-1)
    top_n_indices = top_n_indices.repeat_interleave(key_dim, -1)
    return top_n_indices
