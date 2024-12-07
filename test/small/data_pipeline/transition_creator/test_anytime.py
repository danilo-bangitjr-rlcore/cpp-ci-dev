import numpy as np
import pandas as pd
import datetime

from corerl.utils.torch import tensor_allclose
from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, NewTransition, Step, StageCode, TemporalState
from corerl.data_pipeline.transition_creators.anytime import (
    AnytimeTransitionCreator,
    AnytimeTransitionCreatorConfig,
    _split_at_nans,
    add_one_hot_countdown,
    add_float_countdown,
    add_null_countdown,
)


def get_test_prior_step(state: Tensor, dp=False) -> Step:
    return Step(
        state=state,
        gamma=0,
        action=Tensor([0.]),
        reward=0,
        dp=dp,
    )


def transitions_equal_test(t0: NewTransition, t1: NewTransition):
    return (
            tensor_allclose(t0.prior.state, t1.prior.state)
            and t0.prior.dp == t1.prior.dp
            and t0.post == t1.post
            and t0.n_steps == t1.n_steps
    )


def test_anytime_1():
    """
    Test with a single pipeframe. The tc should construct transitions when the action changes.

    Also tests that the countdown is set to the right value.
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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )
    tc = AnytimeTransitionCreator(cfg, tags)

    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1
    t_0 = transitions[0]

    expected = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10.]),# countdown (last entry) should be 10 steps until decision
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.]),  # countdown (last entry) should now be 9 steps until decision
            action=Tensor([0.]),
            reward=1,
            gamma=0.9,
            dp=False
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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=1,
        countdown='int',
    )
    tc = AnytimeTransitionCreator(cfg, tags)
    transitions = tc(pf).transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 3
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.0]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 9.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2., 8.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 10.]),
            dp=True
        ),
        post=Step(
            state=Tensor([3., 9.]),
            action=Tensor([1.]),
            reward=1.0,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)


def test_anytime_3_action_change():
    """
    Test with a single pipeframe. The tc should construct transitions when the action changes at first and then
    when the decision window is done.
    """
    state_col = np.arange(8)
    cols = {"state": state_col, "action": [0, 0, 1, 1, 1, 2, 2, 3], "reward": [1, 1, 1, 1, 1, 1, 1, 1]}

    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(8)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=3,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )

    tc = AnytimeTransitionCreator(cfg, tags)
    transitions = tc(pf).transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 6

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 3]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 2]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 3]),
            dp=True,
        ),
        post=Step(
            state=Tensor([4., 3]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
            dp=True,
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 2]),
            dp=False,
        ),
        post=Step(
            state=Tensor([4., 3]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=True,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([3., 1.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([4., 3]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=True,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)

    t_4 = transitions[4]
    expected_4 = NewTransition(
        prior=get_test_prior_step(
            Tensor([4., 3.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([6., 1]),
            action=Tensor([2.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_4, expected_4)

    t_5 = transitions[5]
    expected_5 = NewTransition(
        prior=get_test_prior_step(
            Tensor([5., 2.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([6., 1]),
            action=Tensor([2.]),
            reward=1.,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_5, expected_5)


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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=8,
        gamma=0.9,
        n_step=None,
        only_dp_transitions=True,
        countdown='null',
    )

    tc = AnytimeTransitionCreator(cfg, tags)
    transitions = tc(pf).transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([8.]),
            action=Tensor([0.]),
            reward=5.6953279,
            gamma=0.9 ** 8,
            dp=True,
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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )

    tc = AnytimeTransitionCreator(cfg, tags)
    pf = tc(pf)
    assert pf.temporal_state[StageCode.TC] is not None
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.]),
            action=Tensor([0.]),
            reward=1.,
            gamma=0.9,
            dp=False,
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
    )

    pf_2.temporal_state[StageCode.TC] = pf.temporal_state[StageCode.TC]

    transitions = tc(pf_2).transitions
    assert isinstance(transitions, list)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 10]),
            dp=True
        ),
        post=Step(
            state=Tensor([5., 6.]),
            action=Tensor([1.]),
            reward=3.439,
            gamma=0.9 ** 4,
            dp=False,
        ),
        n_steps=4
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 9.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([5., 6.]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
            dp=False
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([3., 8.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([5., 6.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([4., 7.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([5., 6.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )

    tc = AnytimeTransitionCreator(cfg, tags)

    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 4
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9 ** 1,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 10]),
            dp=True,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 9.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([5., 10.]),
            dp=True
        ),
        post=Step(
            state=Tensor([6., 9.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False
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
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]
    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )

    tc = AnytimeTransitionCreator(cfg, tags)
    transitions = tc(pf).transitions
    assert isinstance(transitions, list)

    assert len(transitions) == 4
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9 ** 1,
            dp=False
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 10]),
            dp=True,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 9.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([5., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([6., 9.]),
            action=Tensor([2.]),
            reward=1.,
            gamma=0.9,
            dp=False,
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
        countdown='int',
    )

    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    tc = AnytimeTransitionCreator(cfg, tags)

    tc_ts: TemporalState = {
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
            temporal_state=tc_ts
        )

        pf = tc(pf)
        assert isinstance(pf.transitions, list)
        transitions += pf.transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 3

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 3.]),
            dp=True,

        ),
        post=Step(
            state=Tensor([3., 3.]),
            action=Tensor([1.]),
            reward=2.71,
            gamma=0.9 ** 3,
            dp=True,
        ),
        n_steps=3
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 2.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3., 3.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=True,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 1.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3., 3.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=True
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
        countdown='int',
    )
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    tc = AnytimeTransitionCreator(cfg, tags)

    tc_ts: TemporalState = {
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
            temporal_state=tc_ts
        )

        pf = tc(pf)
        assert isinstance(pf.transitions, list)
        transitions += pf.transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 2

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 5.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([2., 3.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 4.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2., 3.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1,
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
        countdown='int',
    )

    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    tc = AnytimeTransitionCreator(cfg, tags)

    tc_ts: TemporalState = {
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
            temporal_state=tc_ts
        )

        pf = tc(pf)

        assert isinstance(pf.transitions, list)
        transitions += pf.transitions

        if i == 2:
            assert len(pf.transitions) == 2
        else:
            assert len(pf.transitions) == 0

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 2.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([2., 2.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=True,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 1.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2., 2.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=True
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
        countdown='int',
    )

    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    tc = AnytimeTransitionCreator(cfg, tags)

    tc_ts: TemporalState = {
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
            temporal_state=tc_ts
        )

        pf = tc(pf)

        assert isinstance(pf.transitions, list)
        transitions += pf.transitions

        if i == 0:
            assert len(pf.transitions) == 0
        else:
            assert len(pf.transitions) == 1

        tc_ts = pf.temporal_state

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 2.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 1.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 1.]),
            dp=False
        ),
        post=Step(
            state=Tensor([2., 2.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=True
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 2.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([3., 1.]),
            action=Tensor([2.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_2, expected_2)


def test_anytime_online_5():
    """
    Expecting exactly the same output as test_anytime_ts_2_data_gap, but now is online
    """
    tags = [
        TagConfig(
            name='state',
        ),
        TagConfig(
            name='action',
            is_action=True,
        ),
        TagConfig(
            name='reward',
        )
    ]

    cfg = AnytimeTransitionCreatorConfig(
        steps_per_decision=10,
        gamma=0.9,
        n_step=None,
        countdown='int',
    )

    tc = AnytimeTransitionCreator(cfg, tags)

    states = list(range(9))
    actions = [0, 0, 1, 1, np.nan, 1, 1, 0, 0]
    rewards = [1] * 9

    transitions = []
    ts: TemporalState = {
        StageCode.TC: None
    }
    for i in range(9):
        cols = {"state": states[i], "action": actions[i], "reward": rewards[i]}
        dates = [
            datetime.datetime(2024, 1, 1, 1, i)
        ]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)

        pf = PipelineFrame(
            df,
            caller_code=CallerCode.ONLINE,
            temporal_state=ts
        )
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)

        transitions += new_transitions
        ts = pf.temporal_state

        # on the second iteration, feed an extra empty pipeframe thru,
        # this shouldn't affect anything but should raise a warning.
        if i == 1:
            cols = {"state": [], "action": [], "reward": []}
            dates = []
            datetime_index = pd.DatetimeIndex(dates)
            df = pd.DataFrame(cols, index=datetime_index)

            pf = PipelineFrame(
                df,
                caller_code=CallerCode.ONLINE,
                temporal_state=ts
            )
            _ = tc(pf)

    assert len(transitions) == 4
    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([1., 9.]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9 ** 1,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1., 10.]),
            dp=True,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.9,
            gamma=0.9 ** 2,
            dp=False,
        ),
        n_steps=2
    )

    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., 9.]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3., 8.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False,
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([5., 10.]),
            dp=True
        ),
        post=Step(
            state=Tensor([6., 9.]),
            action=Tensor([1.]),
            reward=1.,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )

    assert transitions_equal_test(t_3, expected_3)


def test_split_at_nans_single_nan():
    dates = [datetime.datetime(2024, 1, i) for i in range(1, 6)]
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50]
    }, index=pd.DatetimeIndex(dates))

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
    }, index=pd.DatetimeIndex(dates))

    result = _split_at_nans(df)
    assert len(result) == 1
    df1, has_nan = result[0]
    assert df1.equals(df) and not has_nan


def test_split_at_nans_multiple_nans():
    dates = [datetime.datetime(2024, 1, i) for i in range(1, 7)]
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, np.nan, 6],
        'B': [10, np.nan, 30, 40, 50, 60]
    }, index=pd.DatetimeIndex(dates))

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


def test_one_hot_countdown_1():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_one_hot_countdown(
        rags,
        steps_until_decision=1,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 1., 0.]))


def test_one_hot_countdown_2():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_one_hot_countdown(
        rags,
        steps_until_decision=2,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 0., 1.]))


def test_one_hot_countdown_3():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_one_hot_countdown(
        rags,
        # note that steps_until_decision is now 0, but we get the same output
        # as the above test where it is 2.
        steps_until_decision=0,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 0., 1.]))


def test_float_countdown_1():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_float_countdown(
        rags,
        steps_until_decision=2,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 1.]))


def test_float_countdown_2():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_float_countdown(
        rags,
        steps_until_decision=0,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 1.]))


def test_float_countdown_3():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_float_countdown(
        rags,
        steps_until_decision=1,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0., 0.5]))


def test_null_countdown():
    rags = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=0.,
        gamma=0.,
    )

    rags = add_null_countdown(
        rags,
        steps_until_decision=1,
        steps_per_decision=2)

    assert tensor_allclose(rags.state, Tensor([0.]))
