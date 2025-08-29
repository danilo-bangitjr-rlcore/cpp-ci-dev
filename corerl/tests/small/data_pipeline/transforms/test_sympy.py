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


# ----------------------------------------
# -- Affine Transform Equivalence Tests --
# ----------------------------------------

def test_self_reference():
    """Test that sympy can reference the current tag in expressions."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["input_value", "output"])
    df = pd.DataFrame(
        data=[
            [10.0, 5.0],   # initial output value
            [20.0, 15.0],  # initial output value
            [30.0, 25.0],  # initial output value
        ],
        columns=cols,
        index=idx,
    )

    # Expression that modifies the output based on its current value and input
    # New output = current_output * 2 + input_value
    tf = SympyTransform(SympyConfig(expression="{output} * 2 + {input_value}"))
    tf_data = df.get(["output"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="output",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["output"])
    expected_df = pd.DataFrame(
        data=[
            [5.0 * 2 + 10.0],   # 10 + 10 = 20
            [15.0 * 2 + 20.0],  # 30 + 20 = 50
            [25.0 * 2 + 30.0],  # 50 + 30 = 80
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_basic():
    """Test that sympy can replicate basic affine transform: scale=2, bias=3."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(4)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["x_val", "output"])
    df = pd.DataFrame(
        data=[
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    # Test sympy equivalent of affine with scale=2, bias=3
    # Affine: 2 * x + 3
    tf = SympyTransform(SympyConfig(expression="2 * {x_val} + 3"))
    tf_data = df.get(["output"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="output",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["output"])
    expected_df = pd.DataFrame(
        data=[
            [2.0 * 1.0 + 3.0],  # 5.0
            [2.0 * 2.0 + 3.0],  # 7.0
            [2.0 * 3.0 + 3.0],  # 9.0
            [2.0 * 4.0 + 3.0],  # 11.0
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_scale_only():
    """Test that sympy can replicate affine transform with scale only: scale=0.5, bias=0."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["value", "result"])
    df = pd.DataFrame(
        data=[
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    # Test sympy equivalent of affine with scale=0.5, bias=0
    # Affine: 0.5 * x + 0
    tf = SympyTransform(SympyConfig(expression="0.5 * {value}"))
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
            [0.5 * 10.0],  # 5.0
            [0.5 * 20.0],  # 10.0
            [0.5 * 30.0],  # 15.0
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_bias_only():
    """Test that sympy can replicate affine transform with bias only: scale=1, bias=-5."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(3)]
    idx = pd.DatetimeIndex(dates)

    cols = pd.Index(["temp", "adjusted"])
    df = pd.DataFrame(
        data=[
            [25.0, 0.0],
            [30.0, 0.0],
            [35.0, 0.0],
        ],
        columns=cols,
        index=idx,
    )

    # Test sympy equivalent of affine with scale=1, bias=-5
    # Affine: 1 * x + (-5)
    tf = SympyTransform(SympyConfig(expression="{temp} - 5"))
    tf_data = df.get(["adjusted"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="adjusted",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["adjusted"])
    expected_df = pd.DataFrame(
        data=[
            [25.0 - 5.0],  # 20.0
            [30.0 - 5.0],  # 25.0
            [35.0 - 5.0],  # 30.0
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_normalization():
    """Test sympy equivalent of affine normalization: scale=1/(max-min), bias=-min/(max-min)."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(4)]
    idx = pd.DatetimeIndex(dates)

    # Simulate normalization where original range is [10, 90] -> [0, 1]
    z_min, z_max = 10.0, 90.0
    scale = 1.0 / (z_max - z_min)
    bias = -z_min / (z_max - z_min)

    cols = pd.Index(["raw", "normalized"])
    df = pd.DataFrame(
        data=[
            [10.0, 0.0],  # should normalize to 0
            [30.0, 0.0],  # should normalize to 0.25
            [50.0, 0.0],  # should normalize to 0.5
            [90.0, 0.0],  # should normalize to 1
        ],
        columns=cols,
        index=idx,
    )

    # Test sympy equivalent of normalization affine transform
    # scale * x + bias = (1/(max-min)) * x + (-min/(max-min))
    tf = SympyTransform(SympyConfig(expression=f"{scale} * {{raw}} + {bias}"))
    tf_data = df.get(["normalized"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="normalized",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_cols = pd.Index(["normalized"])
    expected_df = pd.DataFrame(
        data=[
            [0.0],   # (10-10)/(90-10) = 0
            [0.25],  # (30-10)/(90-10) = 20/80 = 0.25
            [0.5],   # (50-10)/(90-10) = 40/80 = 0.5
            [1.0],   # (90-10)/(90-10) = 80/80 = 1.0
        ],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)

def test_abs_function():
    """Test that abs function works correctly in sympy transforms."""
    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(5)]
    idx = pd.DatetimeIndex(dates)

    # Test data around 0.5 to verify absolute value calculation
    cols = pd.Index(["tag-0", "result"])
    df = pd.DataFrame(
        data=[
            [0.2, 0.0],  # abs(0.2 - 0.5) = 0.3
            [0.5, 0.0],  # abs(0.5 - 0.5) = 0.0
            [0.8, 0.0],  # abs(0.8 - 0.5) = 0.3
            [0.1, 0.0],  # abs(0.1 - 0.5) = 0.4
            [0.9, 0.0],  # abs(0.9 - 0.5) = 0.4
        ],
        columns=cols,
        index=idx,
    )

    # Test the expression that was failing before: -Abs({tag-0} - 0.5)
    tf = SympyTransform(SympyConfig(expression="-Abs({tag-0} - 0.5)"))
    tf_data = df.get(["result"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="result",
    )

    # call transform
    tf_carry, _ = tf(tf_carry, ts=None)

    expected_results = [
        -abs(0.2 - 0.5),  # -0.3
        -abs(0.5 - 0.5),  # -0.0
        -abs(0.8 - 0.5),  # -0.3
        -abs(0.1 - 0.5),  # -0.4
        -abs(0.9 - 0.5),  # -0.4
    ]

    expected_cols = pd.Index(["result"])
    expected_df = pd.DataFrame(
        data=[[val] for val in expected_results],
        columns=expected_cols,
        index=idx,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)
