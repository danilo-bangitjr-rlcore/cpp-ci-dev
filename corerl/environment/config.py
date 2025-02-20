from typing import TYPE_CHECKING

from corerl.configs.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class EnvironmentConfig:
    name: str = MISSING
    seed: int = MISSING
    discrete_control: bool = MISSING


    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed
