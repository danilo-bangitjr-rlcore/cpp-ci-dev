import numpy as np
import pandas as pd
import datetime
import math
from torch import Tensor
import torch

from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode
from corerl.data_pipeline.transition_creators.anytime import (
    AnytimeTransitionCreator,
    AnytimeTransitionCreatorConfig,
    _split_at_nans
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
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 1
    t_0 = transitions[0]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1
    assert t_0.post.gamma == 0.9
    assert torch.equal(t_0.post.state, Tensor([1.]))


def test_anytime_2():
    """
    Test with a single pipeframe. Identical test to test_anytime_1 but n_step is now set to one.
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, 0, 0, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['action']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = 1

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 3
    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.0
    assert t_0.post.gamma == 0.9
    assert torch.equal(t_0.post.state, Tensor([1.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([0.]))
    assert t_1.n_steps == 1
    assert t_1.post.reward == 1.0
    assert t_1.post.gamma == 0.9
    assert torch.equal(t_1.post.state, Tensor([2.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 1
    assert t_2.post.reward == 1.0
    assert t_2.post.gamma == 0.9
    assert torch.equal(t_2.post.state, Tensor([3.]))


def test_anytime_3():
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
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 3
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 4

    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]
    t_3 = transitions[3]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.
    assert t_0.post.gamma == 0.9
    assert torch.equal(t_0.post.state, Tensor([1.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 3
    assert t_1.post.reward == 2.71
    assert t_1.post.gamma == 0.9 ** 3
    assert torch.equal(t_1.post.state, Tensor([4.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 2
    assert t_2.post.reward == 1.9
    assert t_2.post.gamma == 0.9 ** 2
    assert torch.equal(t_2.post.state, Tensor([4.]))

    assert torch.equal(t_3.pre.state, Tensor([3.]))
    assert torch.equal(t_3.post.action, Tensor([1.]))
    assert t_3.n_steps == 1
    assert t_3.post.reward == 1.
    assert t_3.post.gamma == 0.9 ** 1
    assert torch.equal(t_3.post.state, Tensor([4.]))


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

    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, tc_ts = tc._inner_call(pf, tc_ts=None)
    assert tc_ts is not None

    assert len(transitions) == 1
    t_0 = transitions[0]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.0
    assert t_0.post.gamma == 0.9 ** 1
    assert torch.equal(t_0.post.state, Tensor([1.]))

    state_col = np.arange(4, 8)
    cols = {"state": state_col, "action": [1, 1, 0, 0], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4, 8)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf_2 = PipelineFrame(df, CallerCode.OFFLINE)
    pf_2.action_tags = ['action']
    pf_2.obs_tags = ['state']
    pf_2.state_tags = ['state']

    transitions, tc_ts = tc._inner_call(pf_2, tc_ts=tc_ts)

    assert len(transitions) == 4
    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]
    t_3 = transitions[3]

    assert torch.equal(t_0.pre.state, Tensor([1.]))
    assert torch.equal(t_0.post.action, Tensor([1.]))
    assert t_0.n_steps == 4
    assert t_0.post.reward == 3.439
    assert  math.isclose(t_0.post.gamma, 0.9 ** 4)
    assert torch.equal(t_0.post.state, Tensor([5.]))

    assert torch.equal(t_1.pre.state, Tensor([2.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 3
    assert t_1.post.reward == 2.71
    assert t_1.post.gamma == 0.9 ** 3
    assert torch.equal(t_1.post.state, Tensor([5.]))

    assert torch.equal(t_2.pre.state, Tensor([3.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 2
    assert t_2.post.reward == 1.9
    assert t_2.post.gamma == 0.9 ** 2
    assert torch.equal(t_2.post.state, Tensor([5.]))

    assert torch.equal(t_3.pre.state, Tensor([4.]))
    assert torch.equal(t_3.post.action, Tensor([1.]))
    assert t_3.n_steps == 1
    assert t_3.post.reward == 1.
    assert t_3.post.gamma == 0.9 ** 1
    assert torch.equal(t_3.post.state, Tensor([5.]))


def test_anytime_4_only_dp():
    """
    Tests making only dp transitions. We will drop the remaining transitions.
    """
    state_col = np.arange(10)
    cols = {"state": state_col, "action": [0] * 10, "reward": [1]*10}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(10)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 8
    cfg.gamma = 0.9
    cfg.n_step = None
    cfg.only_dp_transitions = True

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 1
    t_0 = transitions[0]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 8
    assert math.isclose(t_0.post.reward, 5.6953279)
    assert math.isclose(t_0.post.gamma, 0.9 ** 8)
    assert torch.equal(t_0.post.state, Tensor([8.]))


def test_anytime_ts_2_data_gap():
    """
    Test with a two pipeframes. The tc should NOT use the temporal state from the first pipeframe to construct
    transitions when given the second pipeframe, since there is data gap.
    """

    state_col = np.arange(9)
    cols = {"state": state_col, "action": [0, 0, 1, 1, np.nan, 1, 1, 0, 0], "reward": [1] * 9}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(9)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, tc_ts = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 4

    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]
    t_3 = transitions[3]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.
    assert t_0.post.gamma == 0.9 ** 1
    assert torch.equal(t_0.post.state, Tensor([1.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 2
    assert t_1.post.reward == 1.9
    assert t_1.post.gamma == 0.9 ** 2
    assert torch.equal(t_1.post.state, Tensor([3.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 1
    assert t_2.post.reward == 1.
    assert t_2.post.gamma == 0.9 ** 1
    assert torch.equal(t_2.post.state, Tensor([3.]))

    assert torch.equal(t_3.pre.state, Tensor([5.]))
    assert torch.equal(t_3.post.action, Tensor([1.]))
    assert t_3.n_steps == 1
    assert t_3.post.reward == 1
    assert t_3.post.gamma == 0.9 ** 1
    assert torch.equal(t_3.post.state, Tensor([6.]))


def test_anytime_ts_3_data_gap_with_action_change():
    """
    Test with a two pipeframes. The tc should NOT use the temporal state from the first pipeframe to
    construct transitions when given the second pipeframe,
    since there is data gap. But the action also changes from the first pf to the second.
    """

    state_col = np.arange(9)
    cols = {"state": state_col, "action": [0, 0, 1, 1, np.nan, 2, 2, 0, 0], "reward": [1] * 9}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(9)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df, CallerCode.OFFLINE)
    pf.action_tags = ['action']
    pf.obs_tags = ['state']
    pf.state_tags = ['state']


    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, tc_ts = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 4
    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]
    t_3 = transitions[3]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.
    assert t_0.post.gamma == 0.9 ** 1
    assert torch.equal(t_0.post.state, Tensor([1.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 2
    assert t_1.post.reward == 1.9
    assert t_1.post.gamma == 0.9 ** 2
    assert torch.equal(t_1.post.state, Tensor([3.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 1
    assert t_2.post.reward == 1.
    assert t_2.post.gamma == 0.9 ** 1
    assert torch.equal(t_2.post.state, Tensor([3.]))

    assert torch.equal(t_3.pre.state, Tensor([5.]))
    assert torch.equal(t_3.post.action, Tensor([2.]))
    assert t_3.n_steps == 1
    assert t_3.post.reward == 1
    assert t_3.post.gamma == 0.9 ** 1
    assert torch.equal(t_3.post.state, Tensor([6.]))


def test_anytime_online_1():
    """
    Simulates online mode. Adds actions up to the steps per decision, which triggers creating transitions.
    """
    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 3
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = None

    transitions = []
    for i in range(4):
        cols = {"state": [i], "action": [1], "reward": [1]}
        dates = [datetime.datetime(2024, 1, 1, 1, i)]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(df, CallerCode.ONLINE)
        pf.action_tags = ['action']
        pf.obs_tags = ['state']
        pf.state_tags = ['state']

        new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
        transitions += new_transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 3

    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([1.]))
    assert t_0.n_steps == 3
    assert t_0.post.reward == 2.71
    assert t_0.post.gamma == 0.9 ** 3
    assert torch.equal(t_0.post.state, Tensor([3.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 2
    assert t_1.post.reward == 1.9
    assert t_1.post.gamma == 0.9 ** 2
    assert torch.equal(t_1.post.state, Tensor([3.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([1.]))
    assert t_2.n_steps == 1
    assert t_2.post.reward == 1.0
    assert t_2.post.gamma == 0.9 ** 1
    assert torch.equal(t_2.post.state, Tensor([3.]))


def test_anytime_online_2():
    """
    Simulates online mode. Adds actions up to a change in action, which triggers creating transitions.
    """
    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 5
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = None

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
        pf = PipelineFrame(df, CallerCode.ONLINE)
        pf.action_tags = ['action']
        pf.obs_tags = ['state']
        pf.state_tags = ['state']

        new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
        transitions += new_transitions

        if i != 3:
            assert len(transitions) == 0
        else:
            assert len(transitions) == 2

    t_0 = transitions[0]
    t_1 = transitions[1]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([1.]))
    assert t_0.n_steps == 2
    assert t_0.post.reward == 1.9
    assert t_0.post.gamma == 0.9 ** 2
    assert torch.equal(t_0.post.state, Tensor([2.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 1
    assert t_1.post.reward == 1.
    assert t_1.post.gamma == 0.9 ** 1
    assert torch.equal(t_1.post.state, Tensor([2.]))


def test_anytime_online_3():
    """
    Adds actions until a change of action, which happens after two time steps.
    But this time the steps per decision is also two. So we don't want to add redundant transitions.
    """
    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 2
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = None

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
        pf = PipelineFrame(df, CallerCode.ONLINE)
        pf.action_tags = ['action']
        pf.obs_tags = ['state']
        pf.state_tags = ['state']

        new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
        transitions += new_transitions

        if i == 2:
            assert len(new_transitions) == 2
        else:
            assert len(new_transitions) == 0

    t_0 = transitions[0]
    t_1 = transitions[1]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([1.]))
    assert t_0.n_steps == 2
    assert t_0.post.reward == 1.9
    assert t_0.post.gamma == 0.9 ** 2
    assert torch.equal(t_0.post.state, Tensor([2.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 1
    assert t_1.post.reward == 1.
    assert t_1.post.gamma == 0.9 ** 1
    assert torch.equal(t_1.post.state, Tensor([2.]))


def test_anytime_online_4():
    """
    Same as test_anytime_online_3, but n_step is set to one.
    """
    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 2
    cfg.gamma = 0.9
    cfg.n_step = 1

    tc = AnytimeTransitionCreator(cfg)

    tc_ts = None

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
        pf = PipelineFrame(df, CallerCode.ONLINE)
        pf.action_tags = ['action']
        pf.obs_tags = ['state']
        pf.state_tags = ['state']

        new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
        transitions += new_transitions

        if i == 0:
            assert len(new_transitions) == 0
        else:
            assert len(new_transitions) == 1

    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]

    assert torch.equal(t_0.pre.state, Tensor([0.]))
    assert torch.equal(t_0.post.action, Tensor([1.]))
    assert t_0.n_steps == 1
    assert t_0.post.reward == 1.
    assert t_0.post.gamma == 0.9 ** 1
    assert torch.equal(t_0.post.state, Tensor([1.]))

    assert torch.equal(t_1.pre.state, Tensor([1.]))
    assert torch.equal(t_1.post.action, Tensor([1.]))
    assert t_1.n_steps == 1
    assert t_1.post.reward == 1.
    assert t_1.post.gamma == 0.9 ** 1
    assert torch.equal(t_1.post.state, Tensor([2.]))

    assert torch.equal(t_2.pre.state, Tensor([2.]))
    assert torch.equal(t_2.post.action, Tensor([2.]))
    assert t_2.n_steps == 1
    assert t_2.post.reward == 1.
    assert t_2.post.gamma == 0.9 ** 1
    assert torch.equal(t_2.post.state, Tensor([3.]))


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



