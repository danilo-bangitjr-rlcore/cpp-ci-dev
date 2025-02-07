import datetime
from collections import deque
from math import comb

import numpy as np
import pandas as pd
from torch import Tensor

from corerl.data_pipeline.all_the_time import AllTheTimeTC, AllTheTimeTCConfig, get_n_step_reward
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, Step, Transition
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NullConfig


def make_test_step(i: int, action: float = 0., gamma: float = 0.9, reward: float = 1.0, dp: bool = False) -> Step:
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


def make_pf(start_state: int, end_state: int, ts: dict | None = None) -> PipelineFrame:
    if ts is None:
        ts = dict()
    state_col = np.arange(start_state, end_state)
    length = end_state - start_state
    cols = {"state": state_col}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(start_state, end_state)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        data_mode=DataMode.OFFLINE,
        temporal_state=ts
    )

    # stub out action constructor
    pf.actions = pd.DataFrame({ "action": [0] * length }, index=datetime_index)
    pf.rewards = pd.DataFrame({ "reward": [1] * length }, index=datetime_index)
    return pf


def make_tc(max_n_step: int, min_n_step: int = 1) -> AllTheTimeTC:
    tags = [
        TagConfig(
            name='state',
            is_meta=False,
        ),
        TagConfig(
            name='action',
            action_constructor=[],
            state_constructor=[NullConfig()],
            is_meta=False,
        ),
        TagConfig(
            name='reward',
            is_meta=True,
            state_constructor=[NullConfig()],
        )
    ]

    cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=max_n_step,
        min_n_step=min_n_step,
    )

    tc = AllTheTimeTC(cfg, tags)
    return tc


def test_all_the_time_basic():
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

    expected_0 = Transition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = Transition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_1 == expected_1

    t_2 = transitions[2]
    expected_2 = Transition(
        steps=[
            step_0,
            step_1,
            step_2,
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert t_2 == expected_2


def test_all_the_time_max_n_1():
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
    expected_0 = Transition(
        steps=[
            step_0,
            step_1,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = Transition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_1 == expected_1



def test_all_the_time_ts_1():
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
    expected_0 = Transition(
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
    expected_0 = Transition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_0 == expected_0

    t_1 = transitions[1]
    expected_1 = Transition(
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
    expected_2 = Transition(
        steps=[
            step_2,
            step_3,
        ],
        n_step_reward=1.0,
        n_step_gamma=0.9,
    )
    assert t_2 == expected_2

    t_3 = transitions[3]
    expected_3 = Transition(
        steps=[
            step_1,
            step_2,
            step_3,
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert t_3 == expected_3


def test_all_the_time_online():
    """
    Tests online creation of transitions,
    """
    tc = make_tc(2, 2)

    transitions = []
    ts = {}
    for i in range(5):
        pf = make_pf(i, i+1, ts=ts)
        pf = tc(pf)
        new_transitions = pf.transitions
        assert isinstance(new_transitions, list)
        transitions += new_transitions
        ts = pf.temporal_state

    assert len(transitions) == 3

    expected_0 = Transition(
        steps=[
            make_test_step(0),
            make_test_step(1),
            make_test_step(2),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[0] == expected_0

    expected_1 = Transition(
        steps=[
            make_test_step(1),
            make_test_step(2),
            make_test_step(3),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[1] == expected_1

    expected_2 = Transition(
        steps=[
            make_test_step(2),
            make_test_step(3),
            make_test_step(4),
        ],
        n_step_reward=1.9,
        n_step_gamma=0.81,
    )
    assert transitions[2] == expected_2


def test_all_the_time_data_modes():
    """
    Checks to see if the ts from one caller code is kept separate from the ts of another caller code.
    """
    tc = make_tc(2)

    pf_1 = make_pf(0, 1)
    pf_1.data_mode = DataMode.OFFLINE

    pf_1 = tc(pf_1)
    transitions = pf_1.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0
    ts_1 = pf_1.temporal_state

    pf_2 = make_pf(1, 2)
    pf_2.data_mode = DataMode.ONLINE

    pf_2 = tc(pf_2)
    transitions = pf_2.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 0

    # now, pass in a pf which is a continuation of the first pf
    # make a pf from another caller code
    pf_3 = make_pf(1, 2, ts=ts_1)
    pf_3.data_mode = DataMode.OFFLINE
    pf_3 = tc(pf_3)
    transitions = pf_3.transitions
    assert isinstance(transitions, list)
    assert len(transitions) == 1

    expected_0 = Transition(
        steps=[
            make_test_step(0),
            make_test_step(1),
        ],
        n_step_reward=1.,
        n_step_gamma=0.9,
    )
    assert transitions[0] == expected_0
