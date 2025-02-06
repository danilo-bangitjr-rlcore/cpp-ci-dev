from collections.abc import Iterable

import torch

from corerl.component.optimizers.torch_opts import (
    AdamConfig,
    ArmijoAdamConfig,
    OptimConfig,
    RmspropConfig,
    SgdConfig,
    LSOConfig,
    optim_group,
)

OptimizerConfig = (
    RmspropConfig
    | AdamConfig
    | SgdConfig
    | ArmijoAdamConfig
    | LSOConfig
)

def init_optimizer(
    cfg: OptimConfig,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, param, ensemble)
