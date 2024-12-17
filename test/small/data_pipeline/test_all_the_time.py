import numpy as np
import pandas as pd
import datetime
from collections import deque
from math import comb

from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, NewTransition, Step
from corerl.data_pipeline.all_the_time import (
    get_n_step_reward,
    update_n_step_reward_gamma,
    AllTheTimeTCConfig,
    AllTheTimeTC,
)


def make_test_step(i, action=0., gamma=0.9, reward=1.0, dp=False) -> Step:
    return Step(
        state=Tensor([i]),
        action=Tensor([action]),
        reward=reward,
        gamma=gamma,
        dp=dp,
    )


def test_get_n_step_reward_1():
    q = deque(maxlen=3)

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    q.append(step_0)
    q.append(step_1)
    q.append(step_2)

    n_step_reward, n_step_gamma = get_n_step_reward(q)

    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81


def test_update_n_step_reward_gamma_1():
    q = deque(maxlen=3)

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    q.append(step_0)
    q.append(step_1)
    q.append(step_2)

    n_step_reward, n_step_gamma = None, None

    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_3 = make_test_step(3)

    q.append(step_3)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_4 = make_test_step(4)

    q.append(step_4)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81


def test_update_n_step_reward_gamma_2():
    q = deque(maxlen=3)

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    q.append(step_0)
    q.append(step_1)
    q.append(step_2)

    n_step_reward, n_step_gamma = None, None

    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.81

    step_3 = make_test_step(3, gamma=0.5)

    q.append(step_3)
    n_step_reward, n_step_gamma = update_n_step_reward_gamma(n_step_reward, n_step_gamma, q)
    assert n_step_reward == 1.9
    assert n_step_gamma == 0.45

    step_4 = make_test_step(4, gamma=0.5)

    q.append(step_4)
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

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = NewTransition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_1 == expected_1

    t_2 = transitions[2]
    expected_2 = NewTransition(
        steps=[
            step_0,
            step_1,
            step_2,
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert t_2 == expected_2


def test_all_the_time_2_max_n_1():
    """
    Tests to see if all the time tc will return the correct transitions.
    However, max_n_step is set to 1, so there should only be two transitions
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

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = NewTransition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_1 == expected_1


def test_all_the_time_3_gap():
    """
    Tests to see if all the time tc will return the correct transitions.
    max_n_step is set to 1, so there should only be one transition,
    since there is a datagap following that transition
    """
    state_col = np.arange(4)
    cols = {"state": state_col, "action": [0, 0, np.nan, 0], "reward": [1, 1, 1, 1]}
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
        max_n_step=2,
    )

    tc = AllTheTimeTC(cfg, tags)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0


def test_all_the_time_4_ts():
    """
    Tests to see if the temporal state will be linked between two successive calls to the TC
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

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    # Pass in the next pf, which continues from the above pf
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

    step_2 = make_test_step(2)
    step_3 = make_test_step(3)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = NewTransition(
        steps=[
            step_0,
            step_1,
            step_2,
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert t_1 == expected_1

    t_2 = transitions[2]
    expected_2 = NewTransition(
        steps=[
            step_2,
            step_3,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_2 == expected_2

    t_3 = transitions[3]
    expected_3 = NewTransition(
        steps=[
            step_1,
            step_2,
            step_3,
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert t_3 == expected_3


def test_all_the_time_5_ts():
    """
    Like the above test, not the nan at the end of the first pf should prevent the
    temporal states from being linked
    """
    state_col = np.arange(2)
    cols = {"state": state_col, "action": [0, 0], "reward": [1, np.nan]}
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
    assert len(transitions) == 0

    # Pass in the next pf, which continues from the above pf
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
    assert len(transitions) == 1

    step_2 = make_test_step(2)
    step_3 = make_test_step(3)

    t_0 = transitions[0]
    expected_0 = NewTransition(
        steps=[
            step_2,
            step_3,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0


def test_all_the_time_6_online():
    """
    Tests online creation of transitions
    """

    def _get_pf(i):
        state_col = np.arange(i, i + 1)
        cols = {"state": state_col, "action": [0], "reward": [1]}
        dates = [
            datetime.datetime(2024, 1, 1, 1, j) for j in range(i, i + 1)
        ]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
        )
        return pf

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
        min_n_step=2,
    )
    tc = AllTheTimeTC(cfg, tags)

    transitions = []
    for i in range(10):
        pf = _get_pf(i)
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)
        transitions += new_transitions

    assert len(transitions) == 8

    for i in range(8):
        step_0 = make_test_step(i)
        step_1 = make_test_step(i+1)
        step_2 = make_test_step(i+2)

        expected = NewTransition(
            [step_0, step_1, step_2],
            n_step_reward=1.9,
            n_step_gamma=0.81,
        )
        assert transitions[i] == expected


def test_all_the_time_7_online():
    """
    Like the above test, but iteration 4 has a nan, so should reset the temporal state.
    """

    def _get_pf(i):
        state_col = np.arange(i, i + 1)
        if i == 4:
            cols = {"state": state_col, "action": [np.nan], "reward": [1]}
        else:
            cols = {"state": state_col, "action": [0], "reward": [1]}

        dates = [
            datetime.datetime(2024, 1, 1, 1, j) for j in range(i, i + 1)
        ]
        datetime_index = pd.DatetimeIndex(dates)
        df = pd.DataFrame(cols, index=datetime_index)
        pf = PipelineFrame(
            df,
            caller_code=CallerCode.OFFLINE,
        )
        return pf

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
        min_n_step=2,
    )
    tc = AllTheTimeTC(cfg, tags)

    transitions = []
    for i in range(10):
        pf = _get_pf(i)
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)
        transitions += new_transitions

    assert len(transitions) == 5

    expected_0 = NewTransition(
        steps=[
            make_test_step(0),
            make_test_step(1),
            make_test_step(2),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[0] == expected_0

    expected_1 = NewTransition(
        steps=[
            make_test_step(1),
            make_test_step(2),
            make_test_step(3),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[1] == expected_1

    expected_2 = NewTransition(
        steps=[
            make_test_step(5),
            make_test_step(6),
            make_test_step(7),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[2] == expected_2

    expected_3 = NewTransition(
        steps=[
            make_test_step(6),
            make_test_step(7),
            make_test_step(8),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[3] == expected_3

    expected_4 = NewTransition(
        steps=[
            make_test_step(7),
            make_test_step(8),
            make_test_step(9),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[4] == expected_4
