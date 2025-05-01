from datetime import datetime, timedelta

from corerl.configs.config import config, list_


@config()
class ExperimentConfig:
    """
    An unfortunate catch-all for configurations that have no
    clear home. An effort to rename this config is tracked
    in [PROD-153]
    """
    device: str = 'cpu'
    num_threads: int = 4
    exp_name: str = 'experiment'
    gamma: float = 0.9
    max_steps: int | None = None
    offline_steps: int = 0
    offline_eval_iters: list[int] = list_()
    offline_start_time: datetime | None = None
    offline_end_time: datetime | None = None
    pipeline_batch_duration: timedelta = timedelta(days=7)
    param: str = ''
    param_from_hash: bool = False
    save_path: str = 'output'
    seed: int = 0
    silent: bool = False
    is_simulation: bool = True
