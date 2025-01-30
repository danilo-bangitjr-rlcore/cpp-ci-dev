from collections.abc import Iterable

import torch

from corerl.component.optimizers.torch_opts import (
    AdamConfig,
    ArmijoAdamConfig,
    OptimConfig,
    RmspropConfig,
    SgdConfig,
    optim_group,
)

OptimizerConfig = (
    RmspropConfig
    | AdamConfig
    | SgdConfig
    | ArmijoAdamConfig
)

def init_optimizer(
    cfg: OptimConfig,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, param, ensemble)
