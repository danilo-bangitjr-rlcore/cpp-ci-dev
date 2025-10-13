from typing import TYPE_CHECKING, Any

from lib_config.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class BaseAgentConfig:
    """
    Kind: internal

    Shared configuration between agent types.
    Broken out from GAC for clarity and providing a
    hierarchy between fundamental RL configs and
    GAC-specific configs.
    """
    name: Any = MISSING

    n_updates: int = 1
    """
    Number of algorithm updates per step.
    """

    gamma: float = 0.9
    seed: int = MISSING

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.seed
