from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from corerl.data_pipeline.all_the_time import AllTheTimeTC, AllTheTimeTCConfig
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, Step, Transition
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig, DecisionPointDetector
from corerl.data_pipeline.transition_filter import (
    TransitionFilter,
    TransitionFilterConfig,
    no_nan,
    only_dp,
    only_no_action_change,
    only_post_dp,
)
from test.small.data_pipeline.test_transition_pipeline import pf_from_actions


def make_test_step(
    i: int,
    action: float | list[float] = 0,
    gamma: float = 0.9,
    reward: float = 1.0,
    dp: bool = False,
    ac: bool = False,
) -> Step:
    return Step(
        state=Tensor([i]),
        action=Tensor([action]),
        reward=reward,
        gamma=gamma,
        ac=ac,
        dp=dp,
        action_lo=torch.zeros_like(Tensor([action])),
        action_hi=torch.zeros_like(Tensor([action])),
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
    "actions, expected",
    [
        ([1, 1, 1], True),
        ([1, 0, 0], True), # no action change after second step (after steps[1])
        ([0, 1, 1], True),
        ([0, 1, 0], False), # change between steps[1]/steps[2]
        ([0, 0, 1], False), # change between steps[1]/steps[2]
        ([0, 0, 1, 0], False), # change after steps[1]
        ([0, 0, 0, 1], False), # change after steps[1]
    ]
)
def test_only_no_action_change(actions: list[float], expected: bool):

    pf = pf_from_actions(np.array(actions))
    countdown_cfg = CountdownConfig(
        action_period=timedelta(minutes=10),
        obs_period=timedelta(minutes=1),
        kind='int',
        normalize=False,
    )
    cd_adder = DecisionPointDetector(countdown_cfg)

    # only_no_action_change filter relies on DP detector to detect action changes
    pf = cd_adder(pf)

    # create transitions
    tc_cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=len(actions)-1,
        min_n_step=len(actions)-1,
    )
    tc = AllTheTimeTC(tc_cfg)

    pf.states = pf.data # stub out states for TC
    pf.states = pf.data # stub out states for TC
    pf = tc(pf)

    assert pf.transitions is not None
    assert len(pf.transitions) == 1
    transition = pf.transitions[0]

    assert only_no_action_change(transition) == expected


def test_no_nan():
    """
    Nan checking should ignore nans on the action and reward of the first step
    """
    steps = [
        make_test_step(0, action=np.nan, reward=np.nan),
        make_test_step(1),
        make_test_step(2),
    ]

    transition = Transition(
        steps=steps,
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    assert no_nan(transition)

    # but other nans should be caught
    steps = [
        make_test_step(0, action=np.nan, reward=np.nan),
        make_test_step(1, action=np.nan),
        make_test_step(2),
    ]

    transition = Transition(
        steps=steps,
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    assert not no_nan(transition)

    steps = [
        make_test_step(0, action=np.nan, reward=np.nan),
        make_test_step(1, reward=np.nan),
        make_test_step(2),
    ]

    transition = Transition(
        steps=steps,
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    assert not no_nan(transition)

    steps = [
        make_test_step(0),
        make_test_step(1),
        make_test_step(2),
    ]
    steps[1].state = Tensor([np.nan])

    transition = Transition(
        steps=steps,
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    assert not no_nan(transition)


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

    # no action change and dp is False
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
            make_test_step(2, action=0, dp=True, ac=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    # action change and dp is False
    transition_3 = Transition(
        steps=[
            make_test_step(0, action=0),
            make_test_step(1, action=1),
            make_test_step(2, action=0, dp=False, ac=True),
        ],
        n_step_gamma=0.81,
        n_step_reward=1.9
    )

    df = pd.DataFrame([])

    pf = PipelineFrame(
        df,
        data_mode=DataMode.OFFLINE,
        transitions=[transition_0, transition_1, transition_2, transition_3],
    )

    pf = transition_filter(pf)
    assert pf.transitions is not None
    assert len(pf.transitions) == 1
    assert pf.transitions[0] == transition_0


def test_capture_regular_RL():

    first = make_test_step(
        i=0,
        action=[3,2,1],
        dp=True,
        ac=False
    )
    second = make_test_step(
        i=1,
        action=[4,2,1],
        dp=False,
        ac=True
    )
    intermediate = [make_test_step(i+2, action=[4,2,1]) for i in range(3)]
    last = make_test_step(
        i=5,
        action=[4,2,1],
        dp=True,
        ac=False
    )


    first_transition = Transition(
        steps=[first, second] + intermediate + [last],
        n_step_reward=5, # irrelevant random value
        n_step_gamma=0.8, # irrelevant random value
    )


    first_2 = last
    second_2 = make_test_step(
        i=6,
        action=[5,2,1],
        dp=False,
        ac=True
    )
    intermediate_2 = [make_test_step(i + 7, action=[5,2,1]) for i in range(3)]
    last_2 = make_test_step(
        i=10,
        action=[5,2,1],
        dp=True,
        ac=False
    )

    second_transition = Transition(
        steps=[first_2, second_2] + intermediate_2 + [last_2],
        n_step_reward=5, # irrelevant random value
        n_step_gamma=0.8, # irrelevant random value
    )


    cfg = TransitionFilterConfig(
        filters=[
            'only_pre_dp_or_ac',
            'only_post_dp',
        ],
    )
    transition_filter = TransitionFilter(cfg)

    pf = PipelineFrame(
        pd.DataFrame([]),
        data_mode=DataMode.OFFLINE,
        transitions=[first_transition, second_transition],
    )

    pf = transition_filter(pf)
    assert pf.transitions is not None
    assert len(pf.transitions) == 2
    assert pf.transitions[0] == first_transition
    assert pf.transitions[1] == second_transition
