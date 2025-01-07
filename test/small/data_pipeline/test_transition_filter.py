import pandas as pd
import pytest

from torch import Tensor

from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, Transition, Step
from corerl.data_pipeline.transition_filter import (
    only_dp,
    only_no_action_change,
    only_post_dp,
    TransitionFilterConfig,
    TransitionFilter

)


def make_test_step(i: int, action: float = 0., gamma: float = 0.9, reward: float = 1.0, dp: bool = False) -> Step:
    return Step(
        state=Tensor([i]),
        action=Tensor([action]),
        reward=reward,
        gamma=gamma,
        dp=dp,
    )


def make_test_dp_transition(dps: list[bool]):
    transition = Transition(
        steps=[
            make_test_step(i, dp=dp) for i, dp in enumerate(dps)
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    return transition


@pytest.mark.parametrize(
    "dps, expected",
    [
        ([True, False, True], True),
        ([True, True, False], False),
        ([False, True, False], False),

    ]
)
def test_only_dp_transitions(dps: list[bool], expected: bool):
    transition = make_test_dp_transition(dps)
    assert only_dp(transition) == expected


@pytest.mark.parametrize(
    "dps, expected",
    [
        ([False, True, True], True),
        ([True, False, False], False),
        ([False, True, False], False),

    ]
)
def test_only_post_transitions(dps: list[bool], expected: bool):
    transition = make_test_dp_transition(dps)
    assert only_post_dp(transition) == expected


def make_action_change_transition(
        dps: list[bool],
        actions: list[float]):
    transition = Transition(
        steps=[
            make_test_step(i, dp=dps[i], action=actions[i]) for i in range(len(dps))
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    return transition


@pytest.mark.parametrize(
    "dps, actions, expected",
    [
        ([False, False, False], [0, 1, 0], False),
        ([False, False, False], [0, 0, 0], True),
        ([False, False, False], [1, 0, 0], True),

    ]
)
def test_only_no_action_change(dps: list[bool], actions: list[float], expected: bool):
    transition = make_action_change_transition(dps, actions)
    assert only_no_action_change(transition) == expected


def test_transition_filter_1():
    cfg = TransitionFilterConfig(
        filters=[
            'only_post_dp',
            'only_no_action_change',
        ],
    )
    transition_filter = TransitionFilter(cfg)

    # no action change and dp is true
    transition_0 = Transition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=0),
            make_test_step(2, action=0, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # no action change and dp is True
    transition_1 = Transition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=0),
            make_test_step(2, action=0, dp=False),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # action change and dp is True
    transition_2 = Transition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=1),
            make_test_step(2, action=0, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # action change and dp is False
    transition_3 = Transition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=1),
            make_test_step(2, action=0, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    df = pd.DataFrame([])

    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        transitions=[transition_0, transition_1, transition_2, transition_3],
    )

    pf = transition_filter(pf)
    assert pf.transitions is not None
    assert len(pf.transitions) == 1
    assert pf.transitions[0] == transition_0
