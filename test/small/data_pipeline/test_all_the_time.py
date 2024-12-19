import numpy as np
import pandas as pd
import datetime
from collections import deque
from math import comb

from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode, NewTransition, Step
from corerl.data_pipeline.all_the_time import (
    AllTheTimeTCConfig,
    AllTheTimeTC,
    get_n_step_reward
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


def make_pf(start_state: int, end_state: int, ts=None) -> PipelineFrame:
    if ts is None:
        ts = dict()
    state_col = np.arange(start_state, end_state)
    length = end_state - start_state
    cols = {"state": state_col, "action": [0] * length, "reward": [1] * length}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(start_state, end_state)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        caller_code=CallerCode.OFFLINE,
        temporal_state=ts
    )
    return pf


def make_tc(max_n_step: int, min_n_step: int = 1) -> AllTheTimeTC:
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
        max_n_step=max_n_step,
        min_n_step=min_n_step,
    )

    tc = AllTheTimeTC(cfg, tags)
    return tc


def test_all_the_time_1():
    """
    Tests to see if all the time tc will return the correct transitions.
    """
    tc = make_tc(2)
    pf = make_pf(0, 3)
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
    tc = make_tc(1)
    pf = make_pf(0, 3)
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

    tc = make_tc(2)

    pf = make_pf(0, 4)
    pf.data.loc[pf.data.index[-2], 'action'] = np.nan  # second last action is nan

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

    tc = make_tc(2)
    pf_1 = make_pf(0, 2)
    pf_1 = tc(pf_1)
    transitions = pf_1.transitions
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
    ts_1 = pf_1.temporal_state

    # Pass in the next pf, which continues from the above pf
    pf_2 = make_pf(2, 4, ts=ts_1)

    pf_2 = tc(pf_2)
    transitions = pf_2.transitions
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
    Like the above test, but the nan at the end of the first pf should prevent the
    transitions from the two pfs to be linked
    """
    tc = make_tc(2)
    pf_1 = make_pf(0, 2)
    pf_1.data.loc[pf_1.data.index[-1], 'reward'] = np.nan  # last reward is nan

    pf_1 = tc(pf_1)
    transitions = pf_1.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0

    # Pass in the next pf, which continues from the above pf
    pf_2 = make_pf(2, 4, ts=pf_1.temporal_state)
    pf_2 = tc(pf_2)
    transitions = pf_2.transitions
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

    tc = make_tc(2, 2)

    transitions = []
    ts = {}
    for i in range(10):
        pf = make_pf(i, i+1, ts=ts)
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)
        transitions += new_transitions
        ts = pf.temporal_state

    assert len(transitions) == 8

    for i in range(8):
        step_0 = make_test_step(i)
        step_1 = make_test_step(i + 1)
        step_2 = make_test_step(i + 2)

        expected = NewTransition(
            [step_0, step_1, step_2],
            n_step_reward=1.9,
            n_step_gamma=0.81,
        )
        assert transitions[i] == expected


def test_all_the_time_7_online():
    """
    Like the above test, but iteration i=4 has a nan, so should reset the temporal state.
    """
    tc = make_tc(2, 2)

    transitions = []
    ts = {}
    for i in range(10):
        pf = make_pf(i, i+1, ts=ts)
        if i == 4:
            pf.data.loc[pf.data.index[0], 'action'] = np.nan
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)
        transitions += new_transitions
        ts = pf.temporal_state

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


def test_all_the_time_8_caller_codes():
    """
    Checks to see if the ts from one caller code is kept separate from the ts of another caller code.
    """
    tc = make_tc(2)

    pf_1 = make_pf(0, 1)
    pf_1.caller_code = CallerCode.OFFLINE

    pf_1 = tc(pf_1)
    transitions = pf_1.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0
    ts_1 = pf_1.temporal_state

    pf_2 = make_pf(1, 2)
    pf_2.caller_code = CallerCode.ONLINE

    pf_2 = tc(pf_2)
    transitions = pf_2.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0

    # now, pass in a pf which is a continuation of the first pf
    # make a pf from another caller code
    pf_3 = make_pf(1, 2, ts=ts_1)
    pf_3.caller_code = CallerCode.OFFLINE
    pf_3 = tc(pf_3)
    transitions = pf_3.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    expected_0 = NewTransition(
        steps=[
            make_test_step(0),
            make_test_step(1),
        ],
        n_step_reward=1.,
        n_step_gamma=0.9,
    )
    assert transitions[0] == expected_0
