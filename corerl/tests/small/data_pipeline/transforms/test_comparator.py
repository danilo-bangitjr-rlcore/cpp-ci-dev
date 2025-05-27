import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.comparator import Comparator, ComparatorConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from test.infrastructure.utils.pandas import dfs_close


def test_less_than():
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

    tf = Comparator(ComparatorConfig(op='<', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1<4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [1.0],
            [1.0],
            [1.0],
            [np.nan],
            [0.0],
            [0.0],
        ],
        columns=expected_cols,
        index=idx,
    )
    print(tf_carry.transform_data, expected_df)

    assert dfs_close(tf_carry.transform_data, expected_df)


def test_less_than_equal():
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

    tf = Comparator(ComparatorConfig(op='<=', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1<=4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [1.0],
            [1.0],
            [1.0],
            [np.nan],
            [1.0],
            [0.0],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)

def test_greater_than():
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

    tf = Comparator(ComparatorConfig(op='>', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1>4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0.0],
            [0.0],
            [0.0],
            [np.nan],
            [0.0],
            [1.0],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)

def test_greater_than_equal():
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

    tf = Comparator(ComparatorConfig(op='>=', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1>=4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0.0],
            [0.0],
            [0.0],
            [np.nan],
            [1.0],
            [1.0],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)

def test_equal():
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

    tf = Comparator(ComparatorConfig(op='==', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1==4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0.0],
            [0.0],
            [0.0],
            [np.nan],
            [1.0],
            [0.0],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)

def test_not_equal():
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

    tf = Comparator(ComparatorConfig(op='!=', val=4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-1!=4.0"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [1.0],
            [1.0],
            [1.0],
            [np.nan],
            [0.0],
            [1.0],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)
