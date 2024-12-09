import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.reward.rc import RewardComponentConstructor, RewardConstructor
from corerl.data_pipeline.tag_config import TagConfig
import corerl.data_pipeline.transforms as xforms
from test.infrastructure.utils.pandas import dfs_close


def test_rc1():
    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs_2": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    transform_cfgs = [
        xforms.NormalizerConfig(min=0.0, max=1.0, from_data=False),
        xforms.TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    rc = RewardConstructor(reward_component_constructors)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
            "obs_2_trace-0.1": [1.0, 1.9, 2.89, np.nan, 1.0, 1.9, np.nan],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})
    expected_df = pd.concat((raw_obs, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_null_xform():
    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 2, 3, np.nan, 1, 2],
            "obs_2": [1, 2, 3, np.nan, 1, 2, np.nan],
            "obs_3": [1, 2, 3, np.nan, 1, 2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    transform_cfgs = [
        xforms.NormalizerConfig(min=0.0, max=1.0, from_data=False),
        xforms.TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    # change final xform to null
    reward_component_constructors["obs_3"] = RewardComponentConstructor([xforms.NullConfig()])
    rc = RewardConstructor(reward_component_constructors)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9, 2.89, np.nan, 1.0, 1.9],
            "obs_2_trace-0.1": [1.0, 1.9, 2.89, np.nan, 1.0, 1.9, np.nan],
        }
    )
    expected_reward_vals = expected_components.sum(axis=1, skipna=False)
    expected_reward_df = pd.DataFrame({"reward": expected_reward_vals})
    expected_df = pd.concat((raw_obs, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_lessthan_xform():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xforms.LessThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([True, True, True, False, np.nan, False, False])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_greaterthan_xform():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xforms.GreaterThanConfig(threshold=5),
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([False, False, False, True, np.nan, True, True])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)


def test_greaterthan_penalty_reward():
    """
    This tests the following reward:
        if tag_2 > 5:
            r = -10
        else:
            r = 0
    """
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xforms.GreaterThanConfig(threshold=5),
                xforms.ScaleConfig(factor=-10) # penalty
            ],
        ),
    ]
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [1, 4],
            [2, 6],
            [np.nan, np.nan],
            [4, 10],
            [5, 12],
        ],
        columns=cols,
        index=idx,
    )

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE,
    )

    # call reward constructor
    pf = reward_constructor(pf)

    expected_reward_vals = np.array([0.0, 0.0, 0.0, -10.0, np.nan, -10.0, -10.0])
    expected_reward_df = pd.DataFrame(data=expected_reward_vals, columns=pd.Index(["reward"]), index=idx)
    expected_df = pd.concat((df, expected_reward_df), axis=1, copy=True)

    assert dfs_close(pf.data, expected_df)

def test_null_filter():
    tag_cfgs = [
        TagConfig(
            name="tag-1",
            # default null reward config
        ),
        TagConfig(
            name="tag-2",
            reward_constructor=[
                xforms.GreaterThanConfig(threshold=5),
            ],
        ),
    ]

    reward_components = {cfg.name: RewardComponentConstructor(cfg.reward_constructor) for cfg in tag_cfgs}
    reward_constructor = RewardConstructor(reward_components)

    assert "tag-1" not in reward_constructor.component_constructors
    assert "tag-2" in reward_constructor.component_constructors
