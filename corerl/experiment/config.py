from corerl.configs.config import config


@config()
class ExperimentConfig:
    """
    An unfortunate catch-all for configurations that have no
    clear home. An effort to rename this config is tracked
    in [PROD-153]
    """
    silent: bool = False
