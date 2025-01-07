from typing import Literal
import torch
from collections.abc import Iterable
from corerl.configs.config import config, MISSING
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.component.optimizers.custom_torch_opts import CustomAdam
from corerl.configs.group import Group


optim_group = Group[
    [Iterable[torch.nn.Parameter], bool],
    torch.optim.Optimizer | EnsembleOptimizer,
]()

@config(frozen=True)
class OptimConfig:
    name: str = MISSING

    lr: float = 0.0001
    weight_decay: float = 0.0

def _base_optim(
    optim: type[torch.optim.Adam | torch.optim.SGD | torch.optim.RMSprop | CustomAdam],
    cfg: OptimConfig,
    param: Iterable[torch.nn.Parameter],
    ensemble: bool,
) -> torch.optim.Optimizer | EnsembleOptimizer:
    if not ensemble:
        return optim(param, cfg.lr, weight_decay=cfg.weight_decay)

    return EnsembleOptimizer(
        optim, param,
        kwargs={
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
        },
    )

# -------------
# -- RMSProp --
# -------------
@config(frozen=True)
class RmspropConfig(OptimConfig):
    name: Literal['rms_prop'] = 'rms_prop'


@optim_group.dispatcher
def rmsprop(cfg: RmspropConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.RMSprop,
        cfg, param, ensemble,
    )


# ----------
# -- Adam --
# ----------
@config(frozen=True)
class AdamConfig(OptimConfig):
    name: Literal['adam'] = 'adam'


@optim_group.dispatcher
def adam(cfg: AdamConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.Adam,
        cfg, param, ensemble,
    )


# ----------------
# -- CustomAdam --
# ----------------
@config(frozen=True)
class CustomAdamConfig(OptimConfig):
    name: Literal['custom_adam'] = 'custom_adam'


@optim_group.dispatcher
def custom_adam(cfg: CustomAdamConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        CustomAdam,
        cfg, param, ensemble,
    )


# ---------
# -- SGD --
# ---------
@config(frozen=True)
class SgdConfig(OptimConfig):
    name: Literal['sgd'] = 'sgd'


@optim_group.dispatcher
def sgd(cfg: SgdConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.SGD,
        cfg, param, ensemble,
    )
