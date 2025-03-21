from collections.abc import Iterable
from typing import Any, Literal

import torch

import corerl.component.optimizers.LineSearchOpt.linesearchopt as lso
from corerl.component.optimizers.custom_torch_opts import ArmijoAdam
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.configs.config import MISSING, Field, config
from corerl.configs.group import Group
from corerl.state import AppState

optim_group = Group[
    [AppState, Iterable[torch.nn.Parameter], bool],
    torch.optim.Optimizer | EnsembleOptimizer,
]()

@config()
class OptimConfig:
    name: Any = MISSING

    lr: float = 0.01
    weight_decay: float = 0.0

def _base_optim(
    optim: type[torch.optim.Adam | torch.optim.SGD | torch.optim.RMSprop | ArmijoAdam],
    cfg: OptimConfig,
    app_state : AppState,
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
@config()
class RmspropConfig(OptimConfig):
    name: Literal['rms_prop'] = 'rms_prop'


@optim_group.dispatcher
def rmsprop(cfg: RmspropConfig, app_state: AppState, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.RMSprop,
        cfg, app_state, param, ensemble,
    )


# ----------
# -- Adam --
# ----------
@config()
class AdamConfig(OptimConfig):
    name: Literal['adam'] = 'adam'
    lr: float = 0.001


@optim_group.dispatcher
def adam(cfg: AdamConfig, app_state: AppState, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.Adam,
        cfg, app_state, param, ensemble,
    )


# -------------
# --   LSO   --
# -------------
@config()
class LSOInitConfig:
    name: str = 'To'

@config()
class SearchConditionKwargsConfig:
    c: float = 0.1
    beta: float = 0.9
    min_step_size: float = 0
    max_step_size: float = 1

@config()
class SearchConditionConfig:
    name: str = 'Armijo'
    kwargs: SearchConditionKwargsConfig = Field(default_factory=SearchConditionKwargsConfig)

@config()
class LSOConfig(OptimConfig):
    name: Literal['lso'] = 'lso'
    lr: float = 0.1

    max_backtracking_steps: int = 50
    unit_norm_direction: bool = False
    fallback_step_size: float = 0.0001

    optim: OptimConfig = Field(default_factory=AdamConfig)
    init: LSOInitConfig = Field(default_factory=LSOInitConfig)
    search_condition: SearchConditionConfig = Field(default_factory=SearchConditionConfig)

@optim_group.dispatcher
def lso_dispatch(cfg: LSOConfig, app_state: AppState, param: Iterable[torch.nn.Parameter], ensemble: bool):
    if not cfg.optim.name == 'adam':
        raise ValueError("LSO currently only supports Adam")
    if not ensemble:
        return lso.Optimizer(
            app_state=app_state,
            params=param,
            optim=torch.optim.Adam,
            optim_kwargs={'weight_decay': cfg.optim.weight_decay},
            search_condition=construct_lso_search_condition(cfg.search_condition),
            init=construct_lso_init(cfg.init, cfg.lr),
            init_step_size=cfg.lr,
            max_backtracking_steps=cfg.max_backtracking_steps,
            fallback_step_size=cfg.fallback_step_size,
            unit_norm_direction=cfg.unit_norm_direction
        )

    return EnsembleOptimizer(
        lso.Optimizer, param,
        kwargs={
            "optim": torch.optim.Adam,
            "optim_kwargs": {'weight_decay': cfg.optim.weight_decay},
            "search_condition": construct_lso_search_condition(cfg.search_condition),
            "init": construct_lso_init(cfg.init, cfg.lr),
            "init_step_size": cfg.lr,
            "max_backtracking_steps": cfg.max_backtracking_steps,
            "fallback_step_size": cfg.fallback_step_size,
            "unit_norm_direction": cfg.unit_norm_direction
        },
        app_state=app_state
    )



def construct_lso_init(cfg: LSOInitConfig, lr: float) -> lso.init.StepsizeInit:
    match cfg.name:
        case 'To':
            return lso.init.To(step_size=lr)
        case _:
            raise ValueError("LSO only supports To init condition")

def construct_lso_search_condition(cfg: SearchConditionConfig) -> lso.search.Search:
    kwargs_cfg = cfg.kwargs
    match cfg.name:
        case "Armijo":
            return lso.search.Armijo(
                c=kwargs_cfg.c,
                beta=kwargs_cfg.beta,
                min_step_size=kwargs_cfg.min_step_size,
                max_step_size=kwargs_cfg.max_step_size,
            )
        case _:
            raise ValueError("LSO only supports Armijo search condition")

# ---------
# -- SGD --
# ---------
@config()
class SgdConfig(OptimConfig):
    name: Literal['sgd'] = 'sgd'


@optim_group.dispatcher
def sgd(cfg: SgdConfig, app_state: AppState, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        torch.optim.SGD,
        cfg, app_state, param, ensemble,
    )


# ----------------
# -- ArmijoAdam --
# ----------------
@config()
class ArmijoAdamConfig(OptimConfig):
    name: Literal['armijo_adam'] = 'armijo_adam'
    c: float = 0.1
    tau: float = 0.5
    beta: float = 0.1  # Controls how strict the Armijo condition is
    max_backtracks: int = 10
    min_lr: float = 1e-4


@optim_group.dispatcher
def armijo_adam(cfg: ArmijoAdamConfig, app_state: AppState, param: Iterable[torch.nn.Parameter], ensemble: bool):
    return _base_optim(
        ArmijoAdam,
        cfg, app_state, param, ensemble,
    )
