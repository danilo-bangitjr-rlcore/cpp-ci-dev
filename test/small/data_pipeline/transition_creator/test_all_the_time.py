import numpy as np
import pandas as pd
import datetime
from collections import deque
from math import comb

from corerl.utils.torch import tensor_allclose
from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, NewTransition, Step, StageCode, TemporalState
from corerl.data_pipeline.transition_creators.all_the_time import (
    get_n_step_reward,
    update_n_step_reward_gamma,
    make_transition,
    AllTheTimeTCConfig,
    AllTheTimeTC,
    AllTheTimeTS

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


def test_get_n_step_reward_1():
    q = deque(maxlen=3)

    step_0 = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_1 = Step(
        state=Tensor([1.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_2 = Step(
        state=Tensor([2.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    q.appendleft(step_0)
    q.appendleft(step_1)
    q.appendleft(step_2)

    n_step_reward, n_step_gamma = get_n_step_reward(q)

    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81


def test_update_n_step_reward_gamma_1():
    q = deque(maxlen=3)

    step_0 = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_1 = Step(
        state=Tensor([1.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_2 = Step(
        state=Tensor([2.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    q.appendleft(step_0)
    q.appendleft(step_1)
    q.appendleft(step_2)

    n_step_reward, n_step_gamma = None, None

    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_3 = Step(
        state=Tensor([3.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    q.appendleft(step_3)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_4 = Step(
        state=Tensor([4.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    q.appendleft(step_4)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81


def test_update_n_step_reward_gamma_2():
    q = deque(maxlen=3)

    step_0 = Step(
        state=Tensor([0.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_1 = Step(
        state=Tensor([1.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    step_2 = Step(
        state=Tensor([2.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.9,
        dp=True
    )

    q.appendleft(step_0)
    q.appendleft(step_1)
    q.appendleft(step_2)

    n_step_reward, n_step_gamma = None, None

    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_3 = Step(
        state=Tensor([3.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.5,
        dp=True
    )

    q.appendleft(step_3)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.45

    step_4 = Step(
        state=Tensor([4.]),
        action=Tensor([0.]),
        reward=1.,
        gamma=0.5,
        dp=True
    )

    q.appendleft(step_4)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.5
    assert n_step_gamma == 0.25


def test_all_the_time_1():
    """
    Tests to see if all the time tc will return the correct transitions.
    """
    state_col = np.arange(3)
    cols = {"state": state_col, "action": [0, 0, 0], "reward": [1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(3)
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

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=2,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == comb(3, 2)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2.]),
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
            Tensor([0.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2.]),
            action=Tensor([0.]),
            reward=1.9,
            gamma=0.81,
            dp=False
        ),
        n_steps=2
    )
    assert transitions_equal_test(t_2, expected_2)


def test_all_the_time_2_max_n_1():
    """
    Like the above, but max_n is now set to 1, so we should only have two transitions produced.
    """
    state_col = np.arange(3)
    cols = {"state": state_col, "action": [0, 0, 0], "reward": [1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(3)
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

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=1,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 2

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_1, expected_1)


def test_all_the_time_3_data_gap():
    """
    There is a data gap in the actions, so we will nuke the temporal state and start fresh
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, np.nan, 0, 0], "reward": [1, 1, 1, 1]}
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

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=1,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2., ]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)


def test_all_the_time_4_ts():
    """
    Tests saving and restoring the temporal state.
    """
    state_col = np.arange(2)
    cols = {"state": state_col, "action": [0, 0], "reward": [1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(2)
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

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=2,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., ]),
            dp=False,
        ),
        post=Step(
            state=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    # pass in another pf. The TC should link together the steps here with the previous
    state_col = np.arange(2, 4)
    cols = {"state": state_col, "action": [0, 0], "reward": [1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(2, 4)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
    )
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 4

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([2.]),
            action=Tensor([0.]),
            reward=1.9,
            gamma=0.81,
            dp=False
        ),
        n_steps=2
    )
    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([2.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3]),
            action=Tensor([0.]),
            reward=1,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_2, expected_2)

    t_3 = transitions[3]
    expected_3 = NewTransition(
        prior=get_test_prior_step(
            Tensor([1.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([3.]),
            action=Tensor([0.]),
            reward=1.9,
            gamma=0.81,
            dp=False
        ),
        n_steps=2
    )
    assert transitions_equal_test(t_3, expected_3)


def test_all_the_time_5_ts_datagap():
    """
    Like the above test, but there is a nan value at the start of the second df, which will nuke the temporal state.
    """
    state_col = np.arange(2)
    cols = {"state": state_col, "action": [0, 0], "reward": [1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(2)
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

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=2,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([0., ]),
            dp=False,
        ),
        post=Step(
            state=Tensor([1.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    # pass in another pf. The TC should link together the steps here with the previous
    state_col = np.arange(2, 4)
    cols = {"state": state_col, "action": [np.nan, 0], "reward": [1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(2, 4)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
    )
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0

    # now passing in a new pf will link this one and the previous
    state_col = np.arange(4, 6)
    cols = {"state": state_col, "action": [0, 0], "reward": [1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4, 6)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
    )
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 3

    t_0 = transitions[0]
    expected_0 = NewTransition(
        prior=get_test_prior_step(
            Tensor([3.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([4.]),
            action=Tensor([0.]),
            reward=1.0,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_0, expected_0)

    t_1 = transitions[1]
    expected_1 = NewTransition(
        prior=get_test_prior_step(
            Tensor([4.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([5.]),
            action=Tensor([0.]),
            reward=1.,
            gamma=0.9,
            dp=False
        ),
        n_steps=1
    )
    assert transitions_equal_test(t_1, expected_1)

    t_2 = transitions[2]
    expected_2 = NewTransition(
        prior=get_test_prior_step(
            Tensor([3.,]),
            dp=False,
        ),
        post=Step(
            state=Tensor([5.]),
            action=Tensor([0.]),
            reward=1.9,
            gamma=0.81,
            dp=False
        ),
        n_steps=2
    )
    assert transitions_equal_test(t_2, expected_2)






