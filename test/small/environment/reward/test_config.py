from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.environment.reward.config import RewardConfig


def test_reward_cfg_schema():
    cfg = direct_load_config(
        MainConfig,
        base='test/small/environment/reward/assets',
        config_name='reward_config.yaml',
    )

    assert isinstance(cfg.reward, RewardConfig)
