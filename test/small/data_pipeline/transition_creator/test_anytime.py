import numpy as np
import pandas as pd
import datetime
import math
from torch import Tensor
import torch

from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, transitions_equal, NewTransition, GORAS, StageCode
from corerl.data_pipeline.transition_creators.anytime import (
    AnytimeTransitionCreator,
    AnytimeTransitionCreatorConfig,
    _split_at_nans
)


def get_test_pre_goras(state: Tensor) -> GORAS:
    return GORAS(
        state=state,
        gamma=0,
        obs=state,
        action=Tensor([0.]),
        reward=0,
    )


def transitions_equal_test(t0: NewTransition, t1: NewTransition):
    return (
            t0.pre.state == t1.pre.state
            and t0.post == t1.post
            and t0.n_steps == t1.n_steps
    )


def test_anytime_1():
    """
    Test with a single pipeframe. The tc should construct transitions when the action changes.
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, 0, 1, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state'],
    )

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
    )
    tc = AnytimeTransitionCreator(cfg)

    pf = tc(pf)
    transitions = pf.transitions

    assert len(transitions) == 1
    t_0 = transitions[0]

    expected = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected)


def test_anytime_2_n_step_1():
    """
    Test with a single pipeframe. N_step is now set to one.
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, 0, 0, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state'],
    )
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=1,
    )
    tc = AnytimeTransitionCreator(cfg)
    transitions = tc(pf).transitions

    assert len(transitions) == 3
    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.0,
            gamma=0.9
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)


def test_anytime_3_action_change():
    """
    Test with a single pipeframe. The tc should construct transitions when the action changes at first and then
    when the decision window is done.
    """
    state_col = np.arange(7)
    cols = {"state": state_col, "action": [0, 0, 1, 1, 1, 2, 2], "reward": [1, 1, 1, 1, 1, 1, 1]}

    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(7)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state'],
    )

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=3,
        gamma=0.9,
        n_step=None,
    )

    tc = AnytimeTransitionCreator(cfg)
    transitions = tc(pf).transitions

    assert len(transitions) == 4

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([4.]),
            obs=Tensor([4.]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([4.]),
            obs=Tensor([4.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        pre=get_test_pre_goras(Tensor([3.])),
        post=GORAS(
            state=Tensor([4.]),
            obs=Tensor([4.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)


def test_anytime_4_only_dp():
    """
    Tests making only dp transitions. We will drop the remaining transitions.
    """
    state_col = np.arange(10)
    cols = {"state": state_col, "action": [0] * 10, "reward": [1] * 10}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(10)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state']
    )

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 8
    cfg.gamma = 0.9
    cfg.n_step = None
    cfg.only_dp_transitions = True

    tc = AnytimeTransitionCreator(cfg)
    transitions = tc(pf).transitions

    assert len(transitions) == 1
    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([8.]),
            obs=Tensor([8.]),
            action=Tensor([0.]),
            reward=5.6953279,
            gamma=0.9 ** 8,
        ),
        n_steps=8
    )

    assert transitions_equal_test(t_0, expected_0)


def test_anytime_ts_1():
    """
    Test with a two pipeframes. The tc should use the temporal state from the first pipeframe to construct transitions
    when given the second pipeframe, since there is a decision window spread over them.
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, 0, 1, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state'],
    )

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
    )

    tc = AnytimeTransitionCreator(cfg)
    pf = tc(pf)
    assert pf.temporal_state[StageCode.TC] is not None
    transitions = pf.transitions

    assert len(transitions) == 1
    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    state_col = np.arange(4, 8)
    cols = {"state": state_col, "action": [1, 1, 0, 0], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4, 8)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf_2 = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state'],
    )
    pf_2.temporal_state[StageCode.TC] = pf.temporal_state[StageCode.TC]

    transitions = tc(pf_2).transitions

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([5.]),
            obs=Tensor([5.]),
            action=Tensor([1.]),
            reward=3.439,
            gamma=0.9 ** 4,
        ),
        n_steps=4
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([5.]),
            obs=Tensor([5.]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([3.])),
        post=GORAS(
            state=Tensor([5.]),
            obs=Tensor([5.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        pre=get_test_pre_goras(Tensor([4.])),
        post=GORAS(
            state=Tensor([5.]),
            obs=Tensor([5.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)


def test_anytime_ts_2_data_gap():
    """
    The tc should NOT use the temporal state from the first half of the pipeframe since there is a data gap
    """

    state_col = np.arange(9)
    cols = {"state": state_col, "action": [0, 0, 1, 1, np.nan, 1, 1, 0, 0], "reward": [1] * 9}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(9)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state']
    )

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
    )

    tc = AnytimeTransitionCreator(cfg)

    pf = tc(pf)
    transitions = pf.transitions

    assert len(transitions) == 4
    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9 ** 1,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        pre=get_test_pre_goras(Tensor([5.])),
        post=GORAS(
            state=Tensor([6.]),
            obs=Tensor([6.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)


def test_anytime_ts_3_data_gap_with_action_change():
    """
    The tc should NOT use the temporal state from the first half of the pipeframe since there is a data gap. There is
    also an action change, so we are testing to see if duplicate transitions are returned.
    """

    state_col = np.arange(9)
    cols = {"state": state_col, "action": [0, 0, 1, 1, np.nan, 2, 2, 0, 0], "reward": [1] * 9}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(9)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        action_tags=['action'],
        obs_tags=['state'],
        state_tags=['state']
    )

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
    )

    tc = AnytimeTransitionCreator(cfg)
    transitions = tc(pf).transitions

    assert len(transitions) == 4
    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9 ** 1,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        pre=get_test_pre_goras(Tensor([5.])),
        post=GORAS(
            state=Tensor([6.]),
            obs=Tensor([6.]),
            action=Tensor([2.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)


def test_anytime_online_1():
    """
    Simulates online mode. Adds actions up to the steps per decision, which triggers creating transitions.
    """
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=3,
        gamma=0.9,
        n_step=None,
    )
    tc = AnytimeTransitionCreator(cfg)

    tc_ts = {
        StageCode.TC: None
    }

    transitions = []
    for i in range(4):
        cols = {"state": [i], "action": [1], "reward": [1]}
        dates = [datetime.datetime(2024, 1, 1, 1, i)]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
            action_tags=['action'],
            obs_tags=['state'],
            state_tags=['state'],
            temporal_state=tc_ts
        )

        pf = tc(pf)
        transitions += pf.transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 3

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_2, expected_2)


def test_anytime_online_2():
    """
    Simulates online mode. Adds actions up to a change in action, which triggers creating transitions.
    """

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=5,
        gamma=0.9,
        n_step=None,
    )

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = {
        StageCode.TC: None
    }

    transitions = []
    for i in range(4):
        if i == 3:
            a = 2
        else:
            a = 1

        cols = {"state": [i], "action": [a], "reward": [1]}
        dates = [datetime.datetime(2024, 1, 1, 1, i)]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
            action_tags=['action'],
            obs_tags=['state'],
            state_tags=['state'],
            temporal_state=tc_ts
        )

        pf = tc(pf)
        transitions += pf.transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 2

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_1, expected_1)


def test_anytime_online_3():
    """
    Adds actions until a change of action, which happens after two time steps.
    But this time the steps per decision is also two. So we don't want to add redundant transitions.
    """
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=2,
        gamma=0.9,
        n_step=None,
    )
    tc = AnytimeTransitionCreator(cfg)

    tc_ts = {
        StageCode.TC: None
    }

    transitions = []
    for i in range(4):
        if i == 3:
            a = 2
        else:
            a = 1

        cols = {"state": [i], "action": [a], "reward": [1]}
        dates = [datetime.datetime(2024, 1, 1, 1, i)]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
            action_tags=['action'],
            obs_tags=['state'],
            state_tags=['state'],
            temporal_state=tc_ts
        )

        pf = tc(pf)
        transitions += pf.transitions

        if i == 2:
            assert len(pf.transitions) == 2
        else:
            assert len(pf.transitions) == 0

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_1, expected_1)


def test_anytime_online_4():
    """
    Same as test_anytime_online_3, but n_step is set to one.
    """
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=2,
        gamma=0.9,
        n_step=1,
    )

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = {
        StageCode.TC: None
    }

    transitions = []
    for i in range(4):
        if i == 3:
            a = 2
        else:
            a = 1

        cols = {"state": [i], "action": [a], "reward": [1]}
        dates = [datetime.datetime(2024, 1, 1, 1, i)]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
            action_tags=['action'],
            obs_tags=['state'],
            state_tags=['state'],
            temporal_state=tc_ts
        )

        pf = tc(pf)
        transitions += pf.transitions

        if i == 0:
            assert len(pf.transitions) == 0
        else:
            assert len(pf.transitions) == 1

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        pre=get_test_pre_goras(Tensor([0.])),
        post=GORAS(
            state=Tensor([1.]),
            obs=Tensor([1]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        pre=get_test_pre_goras(Tensor([1.])),
        post=GORAS(
            state=Tensor([2.]),
            obs=Tensor([2.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        pre=get_test_pre_goras(Tensor([2.])),
        post=GORAS(
            state=Tensor([3.]),
            obs=Tensor([3.]),
            action=Tensor([2.]),
            reward=1.,
            gamma=0.9,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_2, expected_2)


def test_split_at_nans_single_nan():
    dates = [datetime.datetime(2024, 1, i) for i in range(1, 6)]
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50]
    }, index=dates)

    result = _split_at_nans(df)
    assert len(result) == 2
    df1, has_nan1 = result[0]
    df2, has_nan2 = result[1]

    assert len(df1) == 1 and has_nan1
    assert len(df2) == 2 and not has_nan2
    assert df1.index[0] == datetime.datetime(2024, 1, 1)
    assert df2.index[-1] == datetime.datetime(2024, 1, 5)


def test_split_at_nans_no_nans():
    dates = [datetime.datetime(2024, 1, i) for i in range(1, 4)]
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [10, 20, 30]
    }, index=dates)

    result = _split_at_nans(df)
    assert len(result) == 1
    df1, has_nan = result[0]
    assert df1.equals(df) and not has_nan


def test_split_at_nans_multiple_nans():
    dates = [datetime.datetime(2024, 1, i) for i in range(1, 7)]
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, np.nan, 6],
        'B': [10, np.nan, 30, 40, 50, 60]
    }, index=dates)

    result = _split_at_nans(df)
    assert len(result) == 3

    df1, has_nan1 = result[0]
    df2, has_nan2 = result[1]
    df3, has_nan3 = result[2]

    assert len(df1) == 1 and has_nan1
    assert len(df2) == 2 and has_nan2
    assert len(df3) == 1 and not has_nan3
    assert df1.index[0] == datetime.datetime(2024, 1, 1)
    assert df2.index[0] == datetime.datetime(2024, 1, 3)
    assert df3.index[0] == datetime.datetime(2024, 1, 6)
