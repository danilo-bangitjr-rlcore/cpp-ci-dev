from math import floor

import torch

from corerl.agent.greedy_ac import grab_percentile, sample_actions


def test_grab_percentile_1():
    values = torch.tensor([[0.1, 0.4, 0.3, 0.2],
                           [0.5, 0.2, 0.1, 0.7],
                           [0.1, 0.2, 0.3, 1],
                           ])

    keys = torch.tensor([
       # first element of batch
       [[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]],

       # second element of batch
       [[9,  10],
        [11, 12],
        [13, 14],
        [15, 16]],

        # third element of batch
       [[17, 18],
        [19, 20],
        [21, 22],
        [23, 24]]])

    percentile = 0.5
    top_keys = grab_percentile(values, keys, percentile)

    expected_top_keys = torch.tensor([
        [[3, 4],
         [5, 6]],

        [[15, 16],
         [9,  10]],

        [[23, 24],
         [21, 22],
        ]])

    assert top_keys.shape == expected_top_keys.shape, "Shape mismatch"
    assert torch.equal(top_keys, expected_top_keys), "Values mismatch"


def test_grab_percentile_2():
    values = torch.tensor([[0.1, 0.4, 0.3, 0.2],
                           [0.5, 0.2, 0.1, 0.7],
                           [0.1, 0.2, 0.3, 1],
                           ])

    keys = torch.tensor([
       # first element of batch
       [[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]],

       # second element of batch
       [[9,  10],
        [11, 12],
        [13, 14],
        [15, 16]],

        # third element of batch
       [[17, 18],
        [19, 20],
        [21, 22],
        [23, 24]]])

    percentile = 0.25
    top_keys = grab_percentile(values, keys, percentile)

    expected_top_keys = torch.tensor([
        [[3, 4]],

        [[15, 16]],

        [[23, 24]]])

    assert top_keys.shape == expected_top_keys.shape, "Shape mismatch"
    assert torch.equal(top_keys, expected_top_keys), "Values mismatch"


class MockActor():
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def get_action(self, state_batch: torch.Tensor, with_grad:bool=False):
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
        policy, # type: ignore
        n_samples,
        action_dim,
        uniform_weight
    )

    assert sampled_actions.shape == (batch_size, n_samples, action_dim)
    assert repeated_states.shape == (batch_size, n_samples, state_dim)
    assert (sampled_actions >= 0).all() and (sampled_actions <= 1).all()

    # the second axis of repeated_states should just be copys of the same state
    assert_rows_identical(repeated_states)
    policy_weight = 1 - uniform_weight
    n_samples_policy = floor(policy_weight * n_samples) # number of samples from the policy
    # the first n_samples_policy actions in sampled_actions should be the same
    assert_rows_identical(sampled_actions[:, :n_samples_policy, :])



