from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config

from corerl.config import MainConfig


def test_offline_config_time_assertion():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/config/offline_config.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert "offline" in cfg.meta
