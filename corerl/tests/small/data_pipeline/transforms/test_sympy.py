import pandas as pd
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config_from_yaml
from lib_utils.list import find, find_instance

from corerl.config import MainConfig
from corerl.data_pipeline.transforms import SympyConfig
from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.sympy import SympyTransform
from tests.sdk.factories import PipelineFrameFactory


def test_simple_addition():
    """Test a simple addition expression with two tags."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [1.0, 3.0, 5.0, 7.0, 9.0],
            "tag-2": [2.0, 4.0, 6.0, 8.0, 10.0],
            "tag-3": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{tag-1} + {tag-2}"))
    tf_data = df.get(["tag-3"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-3",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "tag-3": [3.0, 7.0, 11.0, 15.0, 19.0],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_multiplication_with_constant():
    """Test multiplication with a constant."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [2.0, 4.0, 6.0],
            "tag-2": [0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="3 * {tag-1}"))
    tf_data = df.get(["tag-2"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-2",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "tag-2": [6.0, 12.0, 18.0],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_complex_expression():
    """Test a more complex expression with multiple operations."""
    df = PipelineFrameFactory.build(
        data={
            "pressure": [100.0, 200.0, 150.0],
            "temperature": [300.0, 400.0, 350.0],
            "result": [0.0, 0.0, 0.0],
        },
    ).data

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

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "result": [
                100.0/300.0 + 0.5,
                200.0/400.0 + 0.5,
                150.0/350.0 + 0.5,
            ],
        },
        index=df.index,
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

    df = PipelineFrameFactory.build(
        data={
            "tag-1": [1.0, 2.0, 3.0],
            "tag-2": [3.0, 5.0, 7.0],
            "fake_tag": [2.0, 4.0, 6.0],
        },
    ).data

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
        data={
            "fake_tag": [
                2.0 * 3.0,
                4.0 * 5.0,
                6.0 * 7.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


# ----------------------------------------
# -- Affine Transform Equivalence Tests --
# ----------------------------------------

def test_self_reference():
    """Test that sympy can reference the current tag in expressions."""
    df = PipelineFrameFactory.build(
        data={
            "input_value": [10.0, 20.0, 30.0],
            "output": [5.0, 15.0, 25.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{output} * 2 + {input_value}"))
    tf_data = df.get(["output"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="output",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "output": [
                5.0 * 2 + 10.0,
                15.0 * 2 + 20.0,
                25.0 * 2 + 30.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_basic():
    """Test that sympy can replicate basic affine transform: scale=2, bias=3."""
    df = PipelineFrameFactory.build(
        data={
            "x_val": [1.0, 2.0, 3.0, 4.0],
            "output": [0.0, 0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="2 * {x_val} + 3"))
    tf_data = df.get(["output"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="output",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "output": [
                2.0 * 1.0 + 3.0,
                2.0 * 2.0 + 3.0,
                2.0 * 3.0 + 3.0,
                2.0 * 4.0 + 3.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_scale_only():
    """Test that sympy can replicate affine transform with scale only: scale=0.5, bias=0."""
    df = PipelineFrameFactory.build(
        data={
            "value": [10.0, 20.0, 30.0],
            "result": [0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="0.5 * {value}"))
    tf_data = df.get(["result"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="result",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "result": [
                0.5 * 10.0,
                0.5 * 20.0,
                0.5 * 30.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_bias_only():
    """Test that sympy can replicate affine transform with bias only: scale=1, bias=-5."""
    df = PipelineFrameFactory.build(
        data={
            "temp": [25.0, 30.0, 35.0],
            "adjusted": [0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{temp} - 5"))
    tf_data = df.get(["adjusted"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="adjusted",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "adjusted": [
                25.0 - 5.0,
                30.0 - 5.0,
                35.0 - 5.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_affine_equivalence_normalization():
    """Test sympy equivalent of affine normalization: scale=1/(max-min), bias=-min/(max-min)."""
    z_min, z_max = 10.0, 90.0
    scale = 1.0 / (z_max - z_min)
    bias = -z_min / (z_max - z_min)

    df = PipelineFrameFactory.build(
        data={
            "raw": [10.0, 30.0, 50.0, 90.0],
            "normalized": [0.0, 0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression=f"{scale} * {{raw}} + {bias}"))
    tf_data = df.get(["normalized"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="normalized",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "normalized": [
                0.0,
                0.25,
                0.5,
                1.0,
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)

def test_abs_function():
    """Test that abs function works correctly in sympy transforms."""
    df = PipelineFrameFactory.build(
        data={
            "tag-0": [0.2, 0.5, 0.8, 0.1, 0.9],
            "result": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="-Abs({tag-0} - 0.5)"))
    tf_data = df.get(["result"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="result",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "result": [
                -abs(0.2 - 0.5),
                -abs(0.5 - 0.5),
                -abs(0.8 - 0.5),
                -abs(0.1 - 0.5),
                -abs(0.9 - 0.5),
            ],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_division_basic():
    """Test basic division operation."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [10.0, 15.0, 8.0, 6.0],
            "tag-2": [2.0, 3.0, 4.0, 2.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{tag-1} / {tag-2}"))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    expected_df = pd.DataFrame(
        data={
            "tag-1": [5.0, 5.0, 2.0, 3.0],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(tf_carry.transform_data, expected_df)


def test_division_by_zero_exact():
    """Test division by exact zero produces NaN."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [10.0, 15.0, 8.0],
            "tag-2": [0.0, 2.0, 0.0],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{tag-1} / {tag-2}"))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    results = tf_carry.transform_data["tag-1"].tolist()
    assert pd.isna(results[0])
    assert results[1] == 7.5
    assert pd.isna(results[2])


def test_division_tolerance_behavior():
    """Test division tolerance behavior with small denominators."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [2.0, 2.0, 2.0, 2.0, 2.0],
            "tag-2": [1e-5, 1e-3, 1e-6, 0.1, 1e-4],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{tag-1} / {tag-2}", tolerance=1e-4))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    results = tf_carry.transform_data["tag-1"].tolist()
    assert pd.isna(results[0])
    assert results[1] == 2000.0
    assert pd.isna(results[2])
    assert results[3] == 20.0
    assert pd.isna(results[4])


def test_division_custom_tolerance():
    """Test division with custom tolerance setting."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [2.0, 2.0, 2.0],
            "tag-2": [1e-6, 1e-4, 1e-5],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="{tag-1} / {tag-2}", tolerance=1e-5))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    results = tf_carry.transform_data["tag-1"].tolist()
    assert pd.isna(results[0])
    assert results[1] == 20000.0
    assert pd.isna(results[2])


def test_inverse_expression():
    """Test inverse expression equivalent to old inverse transform."""
    df = PipelineFrameFactory.build(
        data={
            "data": [2.0, 1.0, 0.1, 1e-5, 0.0, -1.0, 0.001],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="1 / {data}", tolerance=1e-4))
    tf_data = df.copy()

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data,
        tag="data",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    results = tf_carry.transform_data["data"].tolist()
    expected = [0.5, 1.0, 10.0, float('nan'), float('nan'), -1.0, 1000.0]

    for i, (result, exp) in enumerate(zip(results, expected, strict=True)):
        if pd.isna(exp):
            assert pd.isna(result), f"Row {i}: expected NaN, got {result}"
        else:
            assert abs(result - exp) < 1e-10, f"Row {i}: expected {exp}, got {result}"


def test_complex_division_expression():
    """Test complex expression with multiple division operations."""
    df = PipelineFrameFactory.build(
        data={
            "tag-1": [12.0, 20.0, 15.0, 18.0],
            "tag-2": [3.0, 4.0, 0.0, 6.0],
            "tag-3": [2.0, 2.0, 3.0, 1e-5],
        },
    ).data

    tf = SympyTransform(SympyConfig(expression="({tag-1} / {tag-2}) / {tag-3}"))
    tf_data = df.get(["tag-1"])
    assert tf_data is not None

    tf_carry = TransformCarry(
        obs=df,
        transform_data=tf_data.copy(),
        tag="tag-1",
    )

    tf_carry, _ = tf(tf_carry, ts=None)

    results = tf_carry.transform_data["tag-1"].tolist()
    assert results[0] == 2.0
    assert results[1] == 2.5
    assert pd.isna(results[2])
    assert pd.isna(results[3])
