import torch
from collections.abc import Iterable
from dataclasses import dataclass
from omegaconf import MISSING
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.component.optimizers.custom_torch_opts import CustomAdam
from corerl.utils.hydra import Group


optim_group = Group[
    [Iterable[torch.nn.Parameter], bool],
    torch.optim.Optimizer | EnsembleOptimizer,
](['agent/critic/critic_optimizer', 'agent/actor/actor_optimizer'])

@dataclass
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
@dataclass
class RmspropConfig(OptimConfig):
    name: str = 'rms_prop'


@optim_group.dispatcher
def _rmsprop(cfg: RmspropConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.RMSprop,
        cfg, param, ensemble,
    )


# ----------
# -- Adam --
# ----------
@dataclass
class AdamConfig(OptimConfig):
    name: str = 'adam'


@optim_group.dispatcher
def _adam(cfg: AdamConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.Adam,
        cfg, param, ensemble,
    )


# ----------------
# -- CustomAdam --
# ----------------
@dataclass
class CustomAdamConfig(OptimConfig):
    name: str = 'custom_adam'


@optim_group.dispatcher
def _custom_adam(cfg: CustomAdamConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        CustomAdam,
        cfg, param, ensemble,
    )


# ---------
# -- SGD --
# ---------
@dataclass
class SgdConfig(OptimConfig):
    name: str = 'sgd'


@optim_group.dispatcher
def _sgd(cfg: SgdConfig, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.SGD,
        cfg, param, ensemble,
    )
