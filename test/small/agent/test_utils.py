
import torch

from corerl.agent.utils import get_sampled_qs, grab_percentile
from corerl.component.network.networks import EnsembleNetworkReturn
from corerl.component.policy_manager import ActionReturn


def test_grab_percentile_multi_key():
    values = torch.tensor(
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.5, 0.2, 0.1, 0.7],
            [0.1, 0.2, 0.3, 1],
        ]
    )

    keys = torch.tensor(
        [
            [[1,  2],  [3,  4],  [5,  6],  [7,  8]],
            [[9,  10], [11, 12], [13, 14], [15, 16]],
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ]
    )

    keys_2 = torch.tensor(
        [
            [[1,  2, 0],  [3,  4, 0],  [5,  6, 0],  [7,  8, 0]],
            [[9,  10, 0], [11, 12, 0], [13, 14, 0], [15, 16, 0]],
            [[17, 18, 0], [19, 20, 0], [21, 22, 0], [23, 24, 0]],
        ]
    )

    percentile = 0.5
    top_keys = grab_percentile(values, [keys, keys_2], percentile)
    assert len(top_keys) == 2

    expected_top_keys_1 = torch.tensor(
        [
            [[3,  4],  [5,  6]],
            [[15, 16], [9,  10]],
            [[23, 24], [21, 22],],
        ]
    )

    assert top_keys[0].shape == expected_top_keys_1.shape, "Shape mismatch"
    assert torch.equal(top_keys[0], expected_top_keys_1), "Values mismatch"

    expected_top_keys_2 = torch.tensor(
        [
            [[3,  4, 0],  [5,  6, 0]],
            [[15, 16, 0], [9,  10, 0]],
            [[23, 24, 0], [21, 22, 0],],
        ]
    )

    assert top_keys[1].shape == expected_top_keys_2.shape, "Shape mismatch"
    assert torch.equal(top_keys[1], expected_top_keys_2), "Values mismatch"


class MockSampler:
    def get_actions(self, states: torch.Tensor, prev_actions: torch.Tensor,):
        policy_actions = torch.ones_like(prev_actions)*.5
        direct_actions = prev_actions + policy_actions
        return ActionReturn(direct_actions, policy_actions)

class MockCritic:
    def get_values(self, states: list[torch.Tensor], actions: list[torch.Tensor]):
        v =states[0][:, 0] + actions[0][:, 0]
        return EnsembleNetworkReturn(v, v)

def test_get_sampled_qs():
    states = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]])

    prev_actions = torch.tensor([
        [0.5, 0.5],
        [0.,  1.]])

    n_samples = 2
    sampler = MockSampler()
    critic = MockCritic()

    result = get_sampled_qs(states, prev_actions, n_samples, sampler.get_actions, critic) # type: ignore

    # equal to state[0] + 0.5 + prev_actions[0]
    expected_q_values = torch.tensor([
        [2.0, 2.0],
        [4.5, 4.5]])

    assert torch.equal(result.q_values, expected_q_values), "Q values mismatch"

    expected_states = torch.tensor([
        [[1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0]],

        [[4.0, 5.0, 6.0],
         [4.0, 5.0, 6.0]]
    ])

    assert torch.equal(result.states, expected_states), "States mismatch"

    expected_direct_actions = torch.tensor([
        [[1.0, 1.0],
         [1.0, 1.0]],

        [[0.5, 1.5],
         [0.5, 1.5]]
    ])

    assert torch.equal(result.direct_actions, expected_direct_actions), "Direct actions mismatch"

    expected_policy_actions = torch.tensor([
        [[0.5, 0.5],
         [0.5, 0.5]],

        [[0.5, 0.5],
         [0.5, 0.5]]
    ])

    assert torch.equal(result.policy_actions, expected_policy_actions), "Policy actions mismatch"
