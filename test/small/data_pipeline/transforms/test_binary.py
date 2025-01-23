import datetime

import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.affine import AffineConfig
from corerl.data_pipeline.transforms.binary import BinaryConfig, BinaryTransform
from corerl.data_pipeline.transforms.interface import TransformCarry
from test.infrastructure.utils.pandas import dfs_close


def test_max():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [4, 1],
            [2, 6],
            [np.nan, np.nan],
            [10, 10],
            [12, 4],
        ],
        columns=cols,
        index=idx,
    )

    tf = BinaryTransform(BinaryConfig(op="max", other="tag-2"))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["max(tag-1, tag-2)"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [2],
            [4],
            [6],
            [np.nan],
            [10],
            [12],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)


def test_max_other_transform():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [4, 1],
            [2, 6],
            [np.nan, np.nan],
            [10, 10],
            [12, 4],
        ],
        columns=cols,
        index=idx,
    )

    tf = BinaryTransform(BinaryConfig(op="max", other="tag-2", other_xform=[AffineConfig(scale=0.5)]))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["max(tag-1, 0.5*tag-2+0.0)"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [1],
            [4],
            [3],
            [np.nan],
            [10],
            [12],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)


def test_min():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [4, 1],
            [2, 6],
            [np.nan, np.nan],
            [10, 10],
            [12, 4],
        ],
        columns=cols,
        index=idx,
    )

    tf = BinaryTransform(BinaryConfig(op="min", other="tag-2"))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["min(tag-1, tag-2)"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0],
            [1],
            [2],
            [np.nan],
            [10],
            [4],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)


def test_min_other_transform():
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [np.nan, 0],
            [0, 2],
            [4, 1],
            [2, 6],
            [np.nan, np.nan],
            [10, 10],
            [12, 4],
        ],
        columns=cols,
        index=idx,
    )

    tf = BinaryTransform(BinaryConfig(op="min", other="tag-2", other_xform=[AffineConfig(scale=0.5)]))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["min(tag-1, 0.5*tag-2+0.0)"])
    expected_df = pd.DataFrame(
        data=[
            [np.nan],
            [0],
            [0.5],
            [2],
            [np.nan],
            [5],
            [2],
        ],
        columns=expected_cols,
        index=idx,
    )

    assert dfs_close(tf_carry.transform_data, expected_df)
