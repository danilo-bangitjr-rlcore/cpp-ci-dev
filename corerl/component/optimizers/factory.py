import torch
from collections.abc import Iterable
from corerl.component.optimizers.torch_opts import optim_group, OptimConfig


def init_optimizer(
    cfg: OptimConfig,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool = False,
):
    return optim_group.dispatch(cfg, param, ensemble)
