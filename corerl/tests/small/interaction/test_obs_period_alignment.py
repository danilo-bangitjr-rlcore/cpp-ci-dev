from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config

from corerl.config import MainConfig


def test_obs_period_shorter_than_action_period():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/interaction/obs_period_longer_than_action_period.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert cfg.message == "Failed to validate config"
    assert "interaction" in cfg.meta
    error = cfg.meta["interaction"]
    assert "The obs_period must be shorter or equal to the action period." in error.message

def test_action_period_divisible_by_obs_period():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/interaction/action_period_not_divisible_by_obs_period.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert cfg.message == "Failed to validate config"
    assert "interaction" in cfg.meta
    error = cfg.meta["interaction"]
    assert "The action_period must be perfectly divisible by the obs_period." in error.message
