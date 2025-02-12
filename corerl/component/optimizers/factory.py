from collections.abc import Iterable

import torch

from corerl.component.optimizers.torch_opts import (
    AdamConfig,
    ArmijoAdamConfig,
    LSOConfig,
    OptimConfig,
    RmspropConfig,
    SgdConfig,
    optim_group,
)
from corerl.state import AppState

OptimizerConfig = (
    RmspropConfig
    | AdamConfig
    | SgdConfig
    | ArmijoAdamConfig
    | LSOConfig
)

def init_optimizer(
    cfg: OptimConfig,
    app_state: AppState,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, app_state, param, ensemble)
