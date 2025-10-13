import os

import pandas as pd
import pytest
from lib_config.errors import ConfigValidationErrors
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.configs.tags.components.bounds import SafetyZonedTag, get_priority_violation_bounds
from corerl.tags.validate_tag_configs import validate_tag_configs
from tests.infrastructure.config import create_config_with_overrides


def load_bounds_config(config_file: str) -> MainConfig:
    """Load a bounds test config and ensure it's valid."""
    config_path = os.path.join(os.path.dirname(__file__), 'assets', config_file)
    cfg = create_config_with_overrides(base_config_path=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    return cfg


def load_validation_config(config_file: str) -> MainConfig:
    """Load a validation test config (expects validation errors)."""
    config_path = os.path.join(os.path.dirname(__file__), 'assets', config_file)
    cfg = create_config_with_overrides(base_config_path=config_path)
    assert not isinstance(cfg, ConfigValidationErrors)
    return cfg


def test_red_bounds_expression():
    cfg = load_bounds_config('bounds_expr.yaml')

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
    cfg = load_bounds_config('bounds_expr.yaml')

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
    cfg = load_validation_config('expected_lower_less_than_op_lower.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_upper_greater_than_op_upper():
    cfg = load_validation_config('expected_upper_greater_than_op_upper.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_upper_less_than_op_lower():
    cfg = load_validation_config('expected_upper_less_than_op_lower.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_expected_lower_greater_than_op_upper():
    cfg = load_validation_config('expected_lower_greater_than_op_upper.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_lower_less_than_op_lower():
    cfg = load_validation_config('red_lower_less_than_op_lower.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_upper_greater_than_op_upper():
    cfg = load_validation_config('red_upper_greater_than_op_upper.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_lower_greater_than_yellow_lower_sympy():
    cfg = load_validation_config('red_lower_greater_than_yellow_lower_sympy.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_yellow_upper_greater_than_red_upper_sympy():
    cfg = load_validation_config('yellow_upper_greater_than_red_upper_sympy.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_zone_reflex_hi_greater_than_reflex_lo():
    cfg = load_validation_config('red_zone_reflex_hi_greater_than_reflex_lo.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_red_zone_reflex_within_op_range():
    cfg = load_validation_config('red_zone_reflex_within_op_range.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_goal_thresh_in_op_range_sympy():
    cfg = load_validation_config('goal_thresh_in_op_range_sympy.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_computed_tag_in_op_range():
    cfg = load_validation_config('computed_tag_in_op_range.yaml')
    with pytest.warns():
        validate_tag_configs(cfg)

def test_redundant_goals():
    cfg = load_validation_config('redundant_goals.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_inconsistent_goals():
    cfg = load_validation_config('inconsistent_goals.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_inconsistent_goals_and_red_zones_float():
    cfg = load_validation_config('inconsistent_goals_and_red_zones_float.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)

def test_inconsistent_goals_and_red_zones_sympy():
    cfg = load_validation_config('inconsistent_goals_and_red_zones_sympy.yaml')
    with pytest.raises(AssertionError):
        validate_tag_configs(cfg)
