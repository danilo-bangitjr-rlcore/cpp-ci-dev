import torch
from collections.abc import Iterable
from corerl.component.optimizers.linesearch_optimizer import LSOConfig
from corerl.component.optimizers.torch_opts import (
    AdamConfig,
    RmspropConfig,
    CustomAdamConfig,
    SgdConfig,
    optim_group,
    OptimConfig,
)


OptimizerConfig = (
    RmspropConfig
    | AdamConfig
    | CustomAdamConfig
    | SgdConfig
    | LSOConfig
)

def init_optimizer(
    cfg: OptimConfig,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, param, ensemble)
