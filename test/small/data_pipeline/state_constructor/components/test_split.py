from datetime import timedelta

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.sc import SCConfig, StateConstructor
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame, StageCode
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import DeltaConfig
from corerl.data_pipeline.transforms.delta import DeltaTemporalState
from corerl.data_pipeline.transforms.split import SplitConfig, SplitTemporalState
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceTemporalState
from test.infrastructure.utils.pandas import dfs_close


def test_split1():
    obs = pd.DataFrame({
        'tag_1': [1, 2, 3, 4, np.nan, 1, 2, 3, 4],
    })

    pf = PipelineFrame(
        data=obs,
        data_mode=DataMode.ONLINE,
    )

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag_1'),
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
        'tag_1_trace-0.1':  [1., 1.9, 2.89, 3.889, np.nan, 1., 1.9, 2.89, 3.889],
        'tag_1_trace-0.01': [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
    })

    assert dfs_close(pf.data, expected_data)


def test_split_ts1():
    obs = pd.DataFrame({
        'tag_1': [1, 2, 3, 4, np.nan, 1, 2, 3, 4],
    })

    ts = SplitTemporalState(
        left_state=[
            TraceTemporalState(
                mu={'tag_1': np.array([100.])},
            ),
            DeltaTemporalState(
                last=np.array([5]),
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

    sc = StateConstructor(
        tag_cfgs=[
            TagConfig(name='tag_1'),
        ],
        cfg=SCConfig(
            defaults=[
                SplitConfig(
                    left=[
                        TraceConfig(trace_values=[0.1]),
                        DeltaConfig(),
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
    expected_data = pd.DataFrame({
        'tag_1_trace-0.1_delta':  [5.9, -8.01, 0.099, 0.9099, np.nan, np.nan, 0.9, 0.99, 0.999],
        'tag_1_trace-0.01': [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
    })

    assert dfs_close(pf.data, expected_data)
