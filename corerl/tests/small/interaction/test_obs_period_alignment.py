from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config

from corerl.config import MainConfig


def test_obs_period_shorter_than_action_period():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/interaction/obs_period_longer_than_action_period.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert "interaction" in cfg.meta

def test_action_period_divisible_by_obs_period():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/interaction/action_period_not_divisible_by_obs_period.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert "interaction" in cfg.meta
