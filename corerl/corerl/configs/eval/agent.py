from lib_config.config import config


@config()
class QOnlineConfig:
    enabled: bool = True


@config()
class GreedDistConfig:
    enabled: bool = True
    n_samples: int = 100


@config()
class QPDFPlotsConfig:
    enabled: bool = True
    primary_action_samples: int = 101
    other_action_samples: int = 10
