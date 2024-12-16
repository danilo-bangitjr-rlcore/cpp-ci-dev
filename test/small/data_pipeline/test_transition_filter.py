import pandas as pd

from torch import Tensor

from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, NewTransition, Step
from corerl.data_pipeline.transition_filter import (
    only_dp,
    only_no_action_change,
    only_post_dp,
    TransitionFilterConfig,
    TransitionFilter

)


def make_test_step(i, action=0., gamma=0.9, reward=1.0, dp=False) -> Step:
    return Step(
        state=Tensor([i]),
        action=Tensor([action]),
        reward=reward,
        gamma=gamma,
        dp=dp,
    )


def test_only_dp_transitions_1():
    transition = NewTransition(
        steps=[
            make_test_step(0, dp=True),
            make_test_step(1),
            make_test_step(2, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert only_dp(transition)


def test_only_dp_transitions_2():
    transition = NewTransition(
        steps=[
            make_test_step(0, dp=True),
            make_test_step(1, True),
            make_test_step(2, dp=False),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert not only_dp(transition)


def test_only_dp_transitions_3():
    transition = NewTransition(
        steps=[
            make_test_step(0, dp=False),
            make_test_step(1, True),
            make_test_step(2, dp=False),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert not only_dp(transition)


def test_only_post_transitions_3():
    transition = NewTransition(
        steps=[
            make_test_step(0, dp=False),
            make_test_step(1, True),
            make_test_step(2, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert only_post_dp(transition)


def test_only_post_transitions_4():
    transition = NewTransition(
        steps=[
            make_test_step(0, dp=True),
            make_test_step(1, False),
            make_test_step(2, dp=False),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert not only_post_dp(transition)


def test_only_no_action_change_1():
    transition = NewTransition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=1),
            make_test_step(2, action=0),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert not only_no_action_change(transition)


def test_only_no_action_change_2():
    transition = NewTransition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=0),
            make_test_step(2, action=0),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert only_no_action_change(transition)


def test_only_no_action_change_3():
    transition = NewTransition(
        steps=[
            make_test_step(0, action=1),
            make_test_step(1, action=0),
            make_test_step(2, action=0),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )
    assert only_no_action_change(transition)


def test_transition_filter_1():
    cfg = TransitionFilterConfig(
        filters=[
            'only_post_dp',
            'only_no_action_change',
        ],
    )
    transition_filter = TransitionFilter(cfg)

    # no action change and dp is true
    transition_0 = NewTransition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=0),
            make_test_step(2, action=0, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # no action change and dp is True
    transition_1 = NewTransition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=0),
            make_test_step(2, action=0, dp=False),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # action change and dp is True
    transition_2 = NewTransition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=1),
            make_test_step(2, action=0, dp=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # action change and dp is False
    transition_3 = NewTransition(
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
    assert isinstance(pf.transitions, list)
    assert len(pf.transitions) == 1
    assert pf.transitions[0] == transition_0
