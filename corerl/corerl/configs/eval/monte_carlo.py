from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class MonteCarloEvalConfig:
    enabled: bool = False
    precision: float = 0.99
    critic_samples: int = 5
    gamma: float = MISSING

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: 'MainConfig'):
        return cfg.agent.gamma
