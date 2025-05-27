from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, StageCode
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import DeltaConfig, NullConfig
from corerl.data_pipeline.transforms.delta import DeltaTemporalState
from corerl.data_pipeline.transforms.split import SplitConfig, SplitTemporalState
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceTemporalState
from test.infrastructure.utils.pandas import dfs_close


def test_split1():
    obs = pd.DataFrame({
        'tag_1':  [1, 2, 3, 4, np.nan, 1, 2, 3, 4],
        'action': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=obs,
        data_mode=DataMode.ONLINE,
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag_1'),
            TagConfig(name='action'),
        ],
        cfg=SCConfig(
            defaults=[
                SplitConfig(
                    left=[TraceConfig(trace_values=[0.1])],
                    right=[TraceConfig(trace_values=[0.01])],
                ),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
            ),
        ),
    )

    pf = sc(pf)
    expected_data = pd.DataFrame({
        'tag_1_trace-0.1':   [1., 1.9, 2.89, 3.889, np.nan, 1., 1.9, 2.89, 3.889],
        'tag_1_trace-0.01':  [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
        'action_trace-0.1':  [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'action_trace-0.01': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'action-lo':         [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'action-hi':         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    assert dfs_close(pf.data, expected_data)


def test_split_ts1():
    obs = pd.DataFrame({
        'tag_1': [1, 2, 3, 4, np.nan, 1, 2, 3, 4],
        'action': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    action_lo = pd.DataFrame({
        'action-lo': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    action_hi = pd.DataFrame({
        'action-hi': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    start_time = datetime(2023, 4, 11, 2)
    increment = timedelta(hours=1)
    timestamps = []
    for i in range(len(obs)):
        timestamps.append(start_time + increment * i)
    obs.index = pd.DatetimeIndex(timestamps)
    action_lo.index = pd.DatetimeIndex(timestamps)
    action_hi.index = pd.DatetimeIndex(timestamps)

    ts = SplitTemporalState(
        left_state=[
            TraceTemporalState(
                mu={'tag_1': np.array([100.])},
            ),
            DeltaTemporalState(
                last=np.array([5.0]),
                time=[start_time - increment]
            ),
        ],
        right_state=None,
    )

    pf = PipelineFrame(
        data=obs,
        data_mode=DataMode.OFFLINE,
        temporal_state={
            StageCode.SC: {
                'tag_1': [ts],
            }
        }
    )
    pf.action_lo = action_lo
    pf.action_hi = action_hi

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag_1'),
            TagConfig(
                name='action',
                state_constructor=[NullConfig()]
            ),
        ],
        cfg=SCConfig(
            defaults=[
                SplitConfig(
                    left=[
                        TraceConfig(trace_values=[0.1]),
                        DeltaConfig(time_thresh=increment),
                    ],
                    right=[TraceConfig(trace_values=[0.01])],
                ),
            ],
            countdown=CountdownConfig(
                action_period=timedelta(minutes=1),
                obs_period=timedelta(minutes=1),
            ),
        ),
    )

    pf = sc(pf)
    print("Got:")
    print(pf.data.to_string())
    expected_data = pd.DataFrame({
        'tag_1_trace-0.1_Δ': [5.9, -8.01, 0.099, 0.9099, np.nan, np.nan, 0.9, 0.99, 0.999],
        'tag_1_trace-0.01':  [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
        'action-lo':         [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'action-hi':         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    assert dfs_close(pf.data, expected_data)
