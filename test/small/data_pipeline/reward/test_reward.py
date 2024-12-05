import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.reward.rc import RewardComponentConstructor, RewardConstructor
from corerl.data_pipeline.state_constructors.components.norm import NormalizerConfig
from corerl.data_pipeline.state_constructors.components.trace import TraceConfig
from corerl.data_pipeline.state_constructors.components.null import NullConfig
from test.infrastructure.utils.pandas import dfs_close


def test_rc1():
    raw_obs = pd.DataFrame(
        {
            "obs_1": [np.nan, 1, 2, 3,      np.nan, 1, 2],
            "obs_2": [1,      2, 3, np.nan, 1,      2, np.nan],
        }
    )

    pf = PipelineFrame(
        data=raw_obs,
        caller_code=CallerCode.REFRESH,
    )

    transform_cfgs = [
        NormalizerConfig(min=0.0, max=1.0, from_data=False),
        TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    rc = RewardConstructor(reward_component_constructors)

    # call reward constructor
    pf = rc(pf)

    expected_components = pd.DataFrame(
        {
            "obs_1_trace-0.1": [np.nan, 1.0, 1.9,  2.89,   np.nan, 1.0, 1.9],
            "obs_2_trace-0.1": [1.0,    1.9, 2.89, np.nan, 1.0,    1.9, np.nan],
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
        NormalizerConfig(min=0.0, max=1.0, from_data=False),
        TraceConfig(trace_values=[0.1]),
    ]
    reward_component_constructors = {
        tag_name: RewardComponentConstructor(transform_cfgs) for tag_name in raw_obs.columns
    }
    # change final xform to null
    reward_component_constructors["obs_3"] = RewardComponentConstructor([NullConfig()])
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
