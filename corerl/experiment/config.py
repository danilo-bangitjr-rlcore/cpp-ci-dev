from datetime import timedelta

from corerl.configs.config import config


@config()
class ExperimentConfig:
    debug: bool = False
    device: str = 'cpu'
    exp_name: str = 'experiment'
    gamma: float = 0.9
    max_steps: int = 200
    offline_steps: int = 0
    pipeline_batch_duration: timedelta = timedelta(days=7)
    render: int = 0
    param: str = ''
    param_from_hash: bool = False
    save_path: str = 'output'
    seed: int = 0
    timeout: int = 1
    run_forever: bool = False
