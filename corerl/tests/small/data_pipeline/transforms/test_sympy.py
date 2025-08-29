import datetime

import pandas as pd
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config_from_yaml
from lib_utils.list import find, find_instance

from corerl.config import MainConfig
from corerl.data_pipeline.transforms import SympyConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.sympy import SympyTransform


def test_simple_addition():
    """Test a simple addition expression with two tags."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(5)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2", "tag-3"])
    df = pd.DataFrame(
        data=[
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 8.0, 0.0],
            [9.0, 10.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    tf = SympyTransform(SympyConfig(expression="{tag-1} + {tag-2}"))
    tf_data = df.get(["tag-3"])  # Transform tag-3, but use tag-1 and tag-2 in expression
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-3",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-3"])
    expected_df = pd.DataFrame(
        data=[
            [3.0],   # 1 + 2
            [7.0],   # 3 + 4
            [11.0],  # 5 + 6
            [15.0],  # 7 + 8
            [19.0],  # 9 + 10
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_multiplication_with_constant():
    """Test multiplication with a constant."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["tag-1", "tag-2"])
    df = pd.DataFrame(
        data=[
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    tf = SympyTransform(SympyConfig(expression="3 * {tag-1}"))
    tf_data = df.get(["tag-2"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-2",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["tag-2"])
    expected_df = pd.DataFrame(
        data=[
            [6.0],   # 3 * 2
            [12.0],  # 3 * 4
            [18.0],  # 3 * 6
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_complex_expression():
    """Test a more complex expression with multiple operations."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["pressure", "temperature", "result"])
    df = pd.DataFrame(
        data=[
            [100.0, 300.0, 0.0],
            [200.0, 400.0, 0.0],
            [150.0, 350.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    tf = SympyTransform(SympyConfig(
        expression="{pressure} / {temperature} + 0.5",
    ))
    tf_data = df.get(["result"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="result",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["result"])
    expected_df = pd.DataFrame(
        data=[
            [100.0/300.0 + 0.5],  # 0.333... + 0.5 = 0.833...
            [200.0/400.0 + 0.5],  # 0.5 + 0.5 = 1.0
            [150.0/350.0 + 0.5],  # 0.428... + 0.5 = 0.928...
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_valid_expression(basic_config_path: str):
    test_cfg = f"""
defaults:
    - {basic_config_path}

pipeline:
    tags:
    - name: fake_tag
      state_constructor:
        - name: sympy
          expression: "{{tag-1}} + 1"
"""
    cfg = direct_load_config_from_yaml(MainConfig, test_cfg)
    assert not isinstance(cfg, ConfigValidationErrors)


def test_invalid_expression(basic_config_path: str):
    """Test that an error is raised for invalid expressions."""

    test_cfg = f"""
defaults:
  - {basic_config_path}

pipeline:
  tags:
    - name: fake_tag
      is_endogenous: True
      state_constructor:
        - name: sympy
          expression: "sin({{fake_tag}})"
"""

    cfg = direct_load_config_from_yaml(MainConfig, test_cfg)
    assert isinstance(cfg, ConfigValidationErrors)
    assert "unsupported operation" in str(cfg.meta).lower()


# ----------------------------
# -- Common Transformations --
# ----------------------------

def test_multiply_two_tags(basic_config_path: str):
    test_cfg = f"""
defaults:
    - {basic_config_path}

pipeline:
    tags:
      - name: fake_tag
        type: default
        state_constructor:
          - name: sympy
            expression: "{{fake_tag}} * {{tag-2}}"
"""

    cfg = direct_load_config_from_yaml(MainConfig, test_cfg)
    assert not isinstance(cfg, ConfigValidationErrors)

    # Create fake data
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)
    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame(
        data=[
            [1.0, 3.0, 2.0],
            [2.0, 5.0, 4.0],
            [3.0, 7.0, 6.0],
        ],
        columns=pd.Index(["tag-1", "tag-2", "fake_tag"]),
        index=idx,
    )

    fake_tag_cfg = find(lambda x: x.name == "fake_tag", cfg.pipeline.tags)
    assert fake_tag_cfg is not None
    assert fake_tag_cfg.state_constructor is not None
    sympy_cfg = find_instance(SympyConfig, fake_tag_cfg.state_constructor)
    assert sympy_cfg is not None

    tf = SympyTransform(sympy_cfg)
    tf_data = df.get(["fake_tag"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="fake_tag",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data=[
            [2.0 * 3.0],
            [4.0 * 5.0],
            [6.0 * 7.0],
        ],
        columns=pd.Index(["fake_tag"]),
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)
