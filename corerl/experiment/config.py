from dataclasses import dataclass
from corerl.utils.hydra import list_

@dataclass
class ExperimentConfig:
    cm_eval_freq: int = 0
    debug: bool = False
    device: str = 'cpu'
    exp_name: str = 'experiment'
    gamma: float = 0.9
    load_env_obs_space_from_data: bool = False
    load_path: str | None = None
    max_steps: int = 200
    offline_stat_keys: list[str] = list_()
    offline_steps: int = 0
    online_learning: bool = True
    online_stat_keys: list[str] = list_()
    online_updates: bool = True
    render: int = 0
    render_after: int = 0
    param: str = ''
    param_from_hash: bool = False
    plot_split: float = 0.999
    plotting: bool = False
    save_path: str = 'output'
    seed: int = 0
    set_env_obs_space: bool = False
    test_epochs: list[int] = list_()
    timeout: int = 1
    train_split: float = 0.9
