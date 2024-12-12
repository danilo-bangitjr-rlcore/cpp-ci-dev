import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame, StageCode
from corerl.data_pipeline.transforms.split import SplitConfig, SplitTemporalState
from corerl.data_pipeline.transforms.trace import TraceConfig, TraceTemporalState
from corerl.data_pipeline.state_constructors.sc import SCConfig, StateConstructor

from corerl.data_pipeline.tag_config import TagConfig
from test.infrastructure.utils.pandas import dfs_close


def test_split1():
    obs = pd.DataFrame({
        'tag_1': [1, 2, 3, 4, np.nan, 1, 2, 3, 4],
    })

    pf = PipelineFrame(
        data=obs,
        caller_code=CallerCode.ONLINE,
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
        left_state=TraceTemporalState(
            mu={
                'tag_1': np.array([100.]),
            },
        ),
        right_state=None,
    )

    pf = PipelineFrame(
        data=obs,
        caller_code=CallerCode.OFFLINE,
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
                    left=[TraceConfig(trace_values=[0.1])],
                    right=[TraceConfig(trace_values=[0.01])],
                ),
            ],
        ),
    )

    pf = sc(pf)
    expected_data = pd.DataFrame({
        'tag_1_trace-0.1':  [10.9, 2.89, 2.989, 3.8989, np.nan, 1.0, 1.9, 2.89, 3.889],
        'tag_1_trace-0.01': [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
    })

    assert dfs_close(pf.data, expected_data)
