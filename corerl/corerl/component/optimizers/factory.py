from collections.abc import Iterable
from typing import Annotated

import torch
from pydantic import Field

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

OptimizerConfig = Annotated[
    RmspropConfig
    | AdamConfig
    | SgdConfig
    | ArmijoAdamConfig
    | LSOConfig
, Field(discriminator='name')]

def init_optimizer(
    cfg: OptimConfig,
    app_state: AppState,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, app_state, param, ensemble)
