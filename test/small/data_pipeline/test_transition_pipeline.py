import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.all_the_time import AllTheTimeTC, AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig, DecisionPointDetector
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.transition_filter import TransitionFilter, TransitionFilterConfig


def pf_from_actions(actions: np.ndarray, ts: dict | None = None) -> PipelineFrame:
    if ts is None:
        ts = dict()

    n = len(actions)
    obs_col = np.random.random(size=n)
    cols = {"obs_0": obs_col}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(n)
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(
        df,
        data_mode=DataMode.OFFLINE,
        temporal_state=ts
    )

    # stub out action constructor
    pf.actions = pd.DataFrame({ "action": actions }, index=datetime_index)
    pf.action_lo = pd.DataFrame({ "action": np.zeros_like(actions) }, index=datetime_index)
    pf.action_hi = pd.DataFrame({ "action": np.ones_like(actions) }, index=datetime_index)
    pf.rewards = pd.DataFrame({ "reward": [1] * n }, index=datetime_index)
    return pf


def test_dp_and_ac_capture():
    obs_period = datetime.timedelta(minutes=1)
    action_period = datetime.timedelta(minutes=3)

    countdown_cfg = CountdownConfig(
        action_period=action_period,
        obs_period=obs_period,
        kind='int',
        normalize=False,
    )
    cd_adder = DecisionPointDetector(countdown_cfg)

    # mark decision points and action change
    action_sequence = np.array([
        1, # step 0, decision point
        2, # step 1, action change
        2, # step 2
        2, # step 3, decision point
        3, # step 4, action change
        3, # step 5
        3, # step 6, decision point
    ])
    pf = pf_from_actions(action_sequence)
    pf = cd_adder(pf)

    assert pf.decision_points[0]
    assert not pf.action_change[0]
    assert pf.data['countdown.[0]'].iloc[0] == 0

    assert not pf.decision_points[1]
    assert pf.action_change[1]
    # countdown should be initialized to steps_per_decision - 1 on action change detection
    assert pf.data['countdown.[0]'].iloc[1] == 2

    assert not pf.decision_points[2]
    assert not pf.action_change[2]
    assert pf.data['countdown.[0]'].iloc[2] == 1

    assert pf.decision_points[3]
    assert not pf.action_change[3]
    assert pf.data['countdown.[0]'].iloc[3] == 0

    assert not pf.decision_points[4]
    assert pf.action_change[4]
    assert pf.data['countdown.[0]'].iloc[4] == 2

    assert not pf.decision_points[5]
    assert not pf.action_change[5]
    assert pf.data['countdown.[0]'].iloc[5] == 1

    assert pf.decision_points[6]
    assert not pf.action_change[6]
    assert pf.data['countdown.[0]'].iloc[6] == 0


def test_regular_rl_capture():
    obs_period = datetime.timedelta(minutes=1)
    action_period = datetime.timedelta(minutes=3)

    # mark decision points and action change
    action_sequence = np.array([
        1, # step 0, decision point
        2, # step 1, action change
        2, # step 2
        2, # step 3, decision point
        3, # step 4, action change
        3, # step 5
        3, # step 6, decision point
        4, # step 7, action change
        4, # step 8
        4, # step 9, decision point
    ])
    pf = pf_from_actions(action_sequence)
    countdown_cfg = CountdownConfig(
        action_period=action_period,
        obs_period=obs_period,
        kind='int',
        normalize=False,
    )

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='obs_0'),
        ],
        cfg=SCConfig(
            defaults=[
                TraceConfig(trace_values=[0.]),
            ],
            countdown=countdown_cfg,
        ),
    )
    pf = sc(pf)

    tc_cfg = AllTheTimeTCConfig(
        gamma=0.9,
        max_n_step=4,
        min_n_step=2,
    )

    tc = AllTheTimeTC(tc_cfg)
    pf = tc(pf)
    transitions = pf.transitions
    assert isinstance(transitions, list)

    cfg = TransitionFilterConfig(
        filters=[
            'only_pre_dp_or_ac',
            'only_post_dp',
            'only_no_action_change',
        ],
    )
    transition_filter = TransitionFilter(cfg)
    pf = transition_filter(pf)

    assert pf.transitions is not None
    assert len(pf.transitions) == 3

    t0 = pf.transitions[0]
    assert len(t0.steps) == 4
    assert t0.steps[0].action == 1
    assert t0.steps[1].action == 2
    assert t0.steps[2].action == 2
    assert t0.steps[3].action == 2

    t1 = pf.transitions[1]
    assert len(t1.steps) == 4
    assert t1.steps[0].action == 2
    assert t1.steps[1].action == 3
    assert t1.steps[2].action == 3
    assert t1.steps[3].action == 3

    t2 = pf.transitions[2]
    assert len(t2.steps) == 4
    assert t2.steps[0].action == 3
    assert t2.steps[1].action == 4
    assert t2.steps[2].action == 4
    assert t2.steps[3].action == 4
