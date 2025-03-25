from typing import TYPE_CHECKING, Any

from corerl.configs.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class EnvironmentConfig:
    name: Any = MISSING
    seed: int = MISSING


    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed
