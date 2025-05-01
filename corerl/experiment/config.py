from corerl.configs.config import config


@config()
class ExperimentConfig:
    """
    An unfortunate catch-all for configurations that have no
    clear home. An effort to rename this config is tracked
    in [PROD-153]
    """
    exp_name: str = 'experiment'
    gamma: float = 0.9
    max_steps: int | None = None
    param: str = ''
    param_from_hash: bool = False
    save_path: str = 'output'
    seed: int = 0
    silent: bool = False
    is_simulation: bool = True
