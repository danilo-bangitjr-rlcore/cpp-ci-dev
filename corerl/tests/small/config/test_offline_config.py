from corerl.config import MainConfig
from lib_config.loader import direct_load_config
from lib_config.errors import ConfigValidationErrors

def test_offline_config_time_assertion():
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/small/config/offline_config.yaml',
    )

    assert isinstance(cfg, ConfigValidationErrors)
    assert cfg.message == "Failed to validate config"
    assert "offline" in cfg.meta
    error = cfg.meta["offline"]
    assert "Offline training start timestamp must come before the offline training end timestamp." in error.message