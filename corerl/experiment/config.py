from corerl.configs.config import config


@config()
class ExperimentConfig:
    """
    An unfortunate catch-all for configurations that have no
    clear home. An effort to rename this config is tracked
    in [PROD-153]
    """
    seed: int = 0
    silent: bool = False
    is_simulation: bool = True
