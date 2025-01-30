import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.conditional_filter import ConditionalFilter
from corerl.data_pipeline.datatypes import DataMode, MissingType, PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import BinaryConfig, ScaleConfig
from corerl.data_pipeline.transforms.comparator import ComparatorConfig
from test.infrastructure.utils.pandas import dfs_close


def test_filter_constructor():
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


    tags=[
        TagConfig(
            name='tag-1',
            # set tag-1 to nan if its value*3 == 6
            filter=[
                ScaleConfig(factor=3),
                ComparatorConfig(op='==', val=6),
            ]
        ),
        TagConfig(name='tag-2', red_bounds=(1.1, 3.3)),
    ]


    pf = PipelineFrame(data=df, data_mode=DataMode.ONLINE)
    filter = ConditionalFilter(tag_cfgs=tags)
    out = filter(pf)

    expected_cols = pd.Index(["tag-1", "tag-2"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0,    2],
            [1,    4],
            [np.nan, 6],
            [np.nan, np.nan],
            [4,      10],
            [5,      12],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(out.data, expected_df)
    expected_missing_vals = [
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.FILTER,  MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
    ]
    expected_missing_df = pd.DataFrame(data=expected_missing_vals, columns=expected_cols, index=idx)
    assert dfs_close(out.missing_info, expected_missing_df)


def test_filter_constructor_with_replace():
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


    tags=[
        TagConfig(
            name='tag-1',
            # set tag-1 to nan if tag-2 == 4
            filter=[
                BinaryConfig(op='replace', other='tag-2'),
                ComparatorConfig(op='==', val=4),
            ]
        ),
        TagConfig(name='tag-2', red_bounds=(1.1, 3.3)),
    ]


    pf = PipelineFrame(data=df, data_mode=DataMode.ONLINE)
    filter = ConditionalFilter(tag_cfgs=tags)
    out = filter(pf)

    expected_cols = pd.Index(["tag-1", "tag-2"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan,  0],
            [0,       2],
            [np.nan,  4],
            [2,       6],
            [np.nan, np.nan],
            [4,      10],
            [5,      12],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(out.data, expected_df)
    expected_missing_vals = [
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.FILTER,  MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
        [MissingType.NULL,    MissingType.NULL],
    ]
    expected_missing_df = pd.DataFrame(data=expected_missing_vals, columns=expected_cols, index=idx)
    assert dfs_close(out.missing_info, expected_missing_df)
