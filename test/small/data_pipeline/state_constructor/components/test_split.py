import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.state_constructors.components.split import SplitConfig
from corerl.data_pipeline.state_constructors.components.trace import TraceConfig
from corerl.data_pipeline.state_constructors.sc import StateConstructor

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
        cfgs=[
            SplitConfig(
                left=TraceConfig(trace_values=[0.1]),
                right=TraceConfig(trace_values=[0.01]),
            ),
        ],
    )

    pf = sc(pf, 'tag_1')
    expected_data = pd.DataFrame({
        'tag_1_trace-0.1':  [1., 1.9, 2.89, 3.889, np.nan, 1., 1.9, 2.89, 3.889],
        'tag_1_trace-0.01': [1., 1.99, 2.9899, 3.989899, np.nan, 1., 1.99, 2.9899, 3.989899],
    })

    assert dfs_close(pf.data, expected_data)
