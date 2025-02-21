from math import floor

import pytest
import torch

from corerl.agent.ac_utils import get_percentile_inds, sample_actions
from corerl.agent.greedy_ac import GreedyAC, GreedyACConfig
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.buffer import MixedHistoryBufferConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.data_pipeline.pipeline import ColumnDescriptions


def test_grab_percentile_1():
    values = torch.tensor(
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.5, 0.2, 0.1, 0.7],
            [0.1, 0.2, 0.3, 1],
        ]
    )

    keys = torch.tensor(
        [
            # first element of batch
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            # second element of batch
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            # third element of batch
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ]
    )

    percentile = 0.5
    top_inds = get_percentile_inds(values, keys, percentile)
    top_keys =  torch.gather(keys, dim=1, index=top_inds)

    expected_top_keys = torch.tensor(
        [
            [[3, 4], [5, 6]],
            [[15, 16], [9, 10]],
            [
                [23, 24],
                [21, 22],
            ],
        ]
    )

    assert top_keys.shape == expected_top_keys.shape, "Shape mismatch"
    assert torch.equal(top_keys, expected_top_keys), "Values mismatch"


def test_grab_percentile_2():
    values = torch.tensor(
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.5, 0.2, 0.1, 0.7],
            [0.1, 0.2, 0.3, 1],
        ]
    )

    keys = torch.tensor(
        [
            # first element of batch
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            # second element of batch
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            # third element of batch
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ]
    )

    percentile = 0.25
    top_inds = get_percentile_inds(values, keys, percentile)
    top_keys =  torch.gather(keys, dim=1, index=top_inds)

    expected_top_keys = torch.tensor([[[3, 4]], [[15, 16]], [[23, 24]]])

    assert top_keys.shape == expected_top_keys.shape, "Shape mismatch"
    assert torch.equal(top_keys, expected_top_keys), "Values mismatch"


class MockActor:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def get_action(self, state_batch: torch.Tensor, with_grad: bool = False):
        batch_size = state_batch.size(0)
        num_samples = state_batch.size(1)
        actions = torch.ones(batch_size, num_samples, self.action_dim)
        return actions, None


def assert_rows_identical(x: torch.Tensor):
    """
    Asserts that for each slice along the first dimension,
    all rows in the 2D matrix are identical.
    """
    assert x.dim() == 3, "Input must be a 3D tensor"

    # Check if all rows in each slice are identical
    assert torch.all(x == x[:, 0:1, :]), "Not all rows are identical in each 2D slice"


def test_sample_actions():
    batch_size = 10
    n_samples = 5
    state_dim = 3
    action_dim = 2
    uniform_weight = 0.5
    policy = MockActor(action_dim)
    state_batch = torch.rand(batch_size, state_dim)

    sampled_actions, repeated_states = sample_actions(
        state_batch,
        n_samples,
        action_dim,
        policy,  # type: ignore
        uniform_weight,
    )

    assert sampled_actions.shape == (batch_size, n_samples, action_dim)
    assert repeated_states.shape == (batch_size, n_samples, state_dim)
    assert (sampled_actions >= 0).all() and (sampled_actions <= 1).all()

    # the second axis of repeated_states should just be copys of the same state
    assert_rows_identical(repeated_states)
    policy_weight = 1 - uniform_weight
    n_samples_policy = floor(policy_weight * n_samples)  # number of samples from the policy
    # the first n_samples_policy actions in sampled_actions should be the same
    assert_rows_identical(sampled_actions[:, :n_samples_policy, :])


class MockCritic:
    def get_q(
        self,
        state_batch: list[torch.Tensor],
        action_batch: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = False,
    ) -> torch.Tensor:
        ab = action_batch[0]
        return ab.max(dim=1)[0]

def assert_n_rows_equal(tensor_1: torch.Tensor, tensor_2: torch.Tensor, n: int):
    """
    Asserts that every consecutive `n` rows in a 2D tensor_1 are identical,
    and equal to a row in tensor_2
    """
    assert tensor_1.dim() == 2 and tensor_2.dim() == 2, "Input tensors must be 2D"
    num_rows_1 = tensor_1.shape[0]
    num_rows_2 = tensor_2.shape[0]

    assert num_rows_1 % n == 0, f"Number of rows ({num_rows_1}) must be a multiple of n ({n})"
    assert num_rows_1 == num_rows_2 * n

    j = 0  # the index in tensor_2
    for i in range(0, num_rows_1, n):
        assert torch.all(
            tensor_2[j] == tensor_1[i : i + n]
        ), f"Rows {i} to {i+n-1} of tensor_1 are not identical to row {i} of tensor_2"
        j += 1


def assert_best_actions(
    action_samples: torch.Tensor,
    best_actions: torch.Tensor,
    batch: int,
    n_samples: int,
    top_n: int,
):
    """
    Verifies that for each batch, `best_actions` contains the top `top_n` actions
    (by maximum element of each action, what mock critic does) selected from `action_samples`.
    """
    # Basic shape checks
    assert action_samples.dim() == 2, "action_samples must be a 2D tensor."
    assert best_actions.dim() == 2, "best_actions must be a 2D tensor."

    expected_samples_rows = batch * n_samples
    expected_best_rows = batch * top_n
    assert (
        action_samples.shape[0] == expected_samples_rows
    ), f"Expected {expected_samples_rows} rows in action_samples, got {action_samples.shape[0]}"
    assert (
        best_actions.shape[0] == expected_best_rows
    ), f"Expected {expected_best_rows} rows in best_actions, got {best_actions.shape[0]}"

    # Process each batch individually
    for b in range(batch):
        # Extract the samples and best actions for the current batch
        samples_batch = action_samples[b * n_samples : (b + 1) * n_samples]
        best_batch = best_actions[b * top_n : (b + 1) * top_n]

        # Compute scores for each action (score = max element in each row)
        sample_scores = samples_batch.max(dim=1).values
        best_scores = best_batch.max(dim=1).values

        # Determine the threshold score:
        # Sort sample scores in descending order. The score at rank `top_n` (zero-indexed: top_n-1)
        # is the minimum score that an action must have to be among the top_n actions.
        sorted_scores, _ = torch.sort(sample_scores, descending=True)
        threshold = sorted_scores[top_n - 1]

        # Check 1: Verify that each best action's score is >= threshold.
        # This ensures that even if there are ties, all best actions meet the minimum score criterion.
        assert torch.all(best_scores >= threshold), (
            f"In batch {b}, found a best action with score below the threshold "
            f"({best_scores} vs threshold {threshold})."
        )

        # Check 2: Verify that each best action is one of the actions in the original samples.
        # For each action in best_batch, ensure that it matches one of the rows in samples_batch.
        for i, best_row in enumerate(best_batch):
            # (samples_batch == best_row) returns a boolean tensor of shape (n_samples, action_dim).
            # .all(dim=1) checks that all elements match for each row.
            matches = (samples_batch == best_row).all(dim=1)
            assert matches.any(), (
                f"In batch {b}, best action at index {i} is not present in the sample actions."
            )
@pytest.mark.parametrize(
    "batch_size, state_dim, action_dim, n_samples, percentile, uniform_weight",
    [
        (4, 3, 2, 5, 0.4, 0.75),
        (6, 4, 3, 10, 0.5, 0.5),
        (8, 5, 4, 7, 0.3, 0.6),
        (8, 5, 4, 7, 0.5, 0.0),
        (100, 10, 10, 100, 0.05, 1.0),
    ],
)
def test_get_top_n_sampled_actions(batch_size:int, state_dim:int, action_dim:int, n_samples:int,
                                   percentile:float, uniform_weight:float):
    cfg = GreedyACConfig(
        actor=NetworkActorConfig(buffer=MixedHistoryBufferConfig(seed=0)),
        critic=EnsembleCriticConfig(buffer=MixedHistoryBufferConfig(seed=0)),
    )
    col_desc = ColumnDescriptions(
        state_cols=[f"tag-{i}" for i in range(state_dim)],
        action_cols=[f"action-{i}" for i in range(action_dim)],
    )
    greedy_ac = GreedyAC(cfg, None, col_desc)  # type: ignore
    greedy_ac.q_critic = MockCritic()  # type: ignore
    greedy_ac.sampler = MockActor(action_dim=action_dim)  # type: ignore

    state_batch = torch.rand(batch_size, state_dim)
    action_batch = torch.rand(batch_size, action_dim)

    top_states, top_actions, sampled_actions, top_direct_actions, sampled_direct_actions = greedy_ac._get_top_n_sampled_actions(  # noqa: E501
        state_batch=state_batch,
        direct_action_batch=action_batch,
        n_samples=n_samples,
        percentile=percentile,
        uniform_weight=uniform_weight,
        sampler=greedy_ac.sampler,
    )

    assert torch.all(top_actions == top_direct_actions)
    assert torch.all(sampled_actions == sampled_direct_actions)

    top_n = floor(n_samples * percentile)
    assert top_states.shape[1] == state_dim
    assert top_actions.shape[1] == action_dim
    assert top_states.shape[0] == int(batch_size * top_n) == top_actions.shape[0]
    assert sampled_actions.shape[0] == batch_size * n_samples
    assert sampled_actions.shape[1] == action_dim

    assert_n_rows_equal(top_states, state_batch, top_n)
    assert_best_actions(sampled_actions, top_actions, batch_size, n_samples, top_n)


@pytest.mark.parametrize(
    "batch_size, state_dim, action_dim, n_samples, percentile, uniform_weight",
    [
        (4, 3, 2, 2, 0.5, 0.75),
        (3, 4, 3, 5, 0.5, 0.5),
        (8, 5, 4, 7, 0.3, 0.6),
        (8, 5, 4, 7, 0.5, 0.0),
        (100, 2, 2, 10, 0.1, 1.0),
    ],
)
def test_get_top_n_sampled_delta_actions(batch_size:int, state_dim:int, action_dim:int, n_samples:int,
                                   percentile:float, uniform_weight:float):
    """
    delta actions version of test_get_top_n_sampled_actions()
    """
    cfg = GreedyACConfig(
        delta_action=True,
        delta_bounds=[(-.1, .1) for _ in range(action_dim)],
        actor=NetworkActorConfig(buffer=MixedHistoryBufferConfig(seed=0)),
        critic=EnsembleCriticConfig(buffer=MixedHistoryBufferConfig(seed=0)),
    )
    col_desc = ColumnDescriptions(
        state_cols=[f"tag-{i}" for i in range(state_dim)],
        action_cols=[f"action-{i}" for i in range(action_dim)] + [f"action-{i}_Δ" for i in range(action_dim)],
    )
    greedy_ac = GreedyAC(cfg, None, col_desc)  # type: ignore
    greedy_ac.q_critic = MockCritic()  # type: ignore
    greedy_ac.sampler = MockActor(action_dim=action_dim)  # type: ignore

    state_batch = torch.rand(batch_size, state_dim)
    action_batch = torch.rand(batch_size, action_dim)

    top_states, top_actions, sampled_actions, top_direct_actions, sampled_direct_actions = greedy_ac._get_top_n_sampled_actions(  # noqa: E501
        state_batch=state_batch,
        direct_action_batch=action_batch,
        n_samples=n_samples,
        percentile=percentile,
        uniform_weight=uniform_weight,
        sampler=greedy_ac.sampler,
    )

    assert torch.any(top_actions != top_direct_actions)
    assert torch.any(sampled_actions != sampled_direct_actions)

    top_n = floor(n_samples * percentile)
    assert top_states.shape[1] == state_dim
    assert top_states.shape[0] == int(batch_size * top_n) == top_actions.shape[0]
    assert top_actions.shape[0] == int(batch_size * top_n)
    assert top_actions.shape[1] == action_dim
    assert sampled_actions.shape[0] == batch_size * n_samples
    assert sampled_actions.shape[1] == action_dim

    assert_n_rows_equal(top_states, state_batch, top_n)
    assert_best_actions(sampled_direct_actions, top_direct_actions, batch_size, n_samples, top_n)
