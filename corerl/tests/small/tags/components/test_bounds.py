import os

import pandas as pd
import pytest
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.tags.components.bounds import SafetyZonedTag, get_priority_violation_bounds
from corerl.tags.validate_tag_configs import validate_tag_configs


def test_red_bounds_expression():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/bounds_expr.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)

    tag2 = (
        Maybe.find(lambda t: t.name == 'tag-2', cfg.pipeline.tags)
        .is_instance(SafetyZonedTag)
        .expect()
    )

    df = pd.DataFrame({'tag-1': [5], 'tag-2': [0]})
    lo, hi = get_priority_violation_bounds(tag2, df)
    assert lo.unwrap() == 4.0
    assert hi.unwrap() == 6.0


def test_yellow_bounds_expression():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/bounds_expr.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)

    tag3 = (
        Maybe.find(lambda t: t.name == 'tag-3', cfg.pipeline.tags)
        .is_instance(SafetyZonedTag)
        .expect()
    )

    df = pd.DataFrame({'tag-1': [7], 'tag-3': [0]})
    lo, hi = get_priority_violation_bounds(tag3, df)
    assert lo.unwrap() == 5.0
    assert hi.unwrap() == 9.0

def test_expected_lower_less_than_op_lower():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/expected_lower_less_than_op_lower.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_upper_greater_than_op_upper():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/expected_upper_greater_than_op_upper.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_upper_less_than_op_lower():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/expected_upper_less_than_op_lower.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_lower_greater_than_op_upper():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/expected_lower_greater_than_op_upper.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_lower_less_than_op_lower():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/red_lower_less_than_op_lower.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_upper_greater_than_op_upper():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/red_upper_greater_than_op_upper.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_lower_greater_than_yellow_lower_sympy():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/red_lower_greater_than_yellow_lower_sympy.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_yellow_upper_greater_than_red_upper_sympy():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/yellow_upper_greater_than_red_upper_sympy.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_goal_thresh_in_op_range_sympy():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'assets/goal_thresh_in_op_range_sympy.yaml',
    )
    cfg = direct_load_config(MainConfig, config_name=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)
