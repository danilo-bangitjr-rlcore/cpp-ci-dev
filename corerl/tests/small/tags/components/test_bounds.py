import os

import pandas as pd
from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from lib_utils.maybe import Maybe

from corerl.config import MainConfig
from corerl.tags.components.bounds import SafetyZonedTag, get_tag_bounds


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
    lo, hi = get_tag_bounds(tag2, df)
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
    lo, hi = get_tag_bounds(tag3, df)
    assert lo.unwrap() == 5.0
    assert hi.unwrap() == 9.0
