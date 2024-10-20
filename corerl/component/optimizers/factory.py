from omegaconf import DictConfig
import linesearchopt as lso
from typing import Optional
import torch
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.component.optimizers.custom_torch_opts import CustomAdam
from typing import Iterator


def init_optimizer(cfg: DictConfig, param: list | dict | Iterator, ensemble: Optional['bool'] = False, vmap: Optional['bool'] = False):
    """
    config files: root/config/agent/critic/critic_optimizer or root/config/agent/actor/actor_optimizer
    """
    name = cfg.name
    if "lr" in cfg.keys():
        lr = cfg.lr
    if "weight_decay" in cfg.keys():
        weight_decay = cfg.weight_decay

    if ensemble and not vmap:
        if name != "lso":
            kwargs = {'weight_decay': cfg.weight_decay, 'lr': cfg.lr}

        if name == "rms_prop":
            return EnsembleOptimizer(torch.optim.RMSprop, param, kwargs=kwargs)
        elif name == 'adam':
            return EnsembleOptimizer(torch.optim.Adam, param, kwargs=kwargs)
        elif name == 'custom_adam':
            return EnsembleOptimizer(CustomAdam, param, kwargs=kwargs)
        elif name == "sgd":
            return EnsembleOptimizer(torch.optim.SGD, param, kwargs=kwargs)
        elif name == "lso":
            kwargs = lso_kwargs(cfg)
            return EnsembleOptimizer(lso.Optimizer, param, kwargs=kwargs)
        else:
            raise NotImplementedError
    else:
        if name == "rms_prop":
            return torch.optim.RMSprop(param, lr, weight_decay=weight_decay)
        elif name == 'adam':
            return torch.optim.Adam(param, lr, weight_decay=weight_decay)
        elif name == 'custom_adam':
            return CustomAdam(param, lr, weight_decay=weight_decay)
        elif name == "sgd":
            return torch.optim.SGD(param, lr, weight_decay=weight_decay)
        elif name == "lso":
            return construct_lso(param, cfg)
        else:
            raise NotImplementedError


def construct_lso(param, cfg):
    kwargs = lso_kwargs(cfg)
    optim = lso.Optimizer(param, **kwargs)
    return optim


def lso_kwargs(cfg):
    kwargs = {}
    init_cfg = cfg.init
    init = construct_lso_init(init_cfg)
    kwargs["init"] = init

    search_cfg = cfg.search_condition
    search_condition = construct_lso_search_condition(search_cfg)
    kwargs["search_condition"] = search_condition

    kwargs["init_step_size"] = cfg["init_step_size"]
    kwargs["max_backtracking_steps"] = cfg["max_backtracking_steps"]

    kwargs["unit_norm_direction"] = cfg.get("unit_norm_direction", False)

    if "fallback_step_size" in cfg.keys():
        kwargs["fallback_step_size"] = cfg["fallback_step_size"]
    else:
        kwargs["fallback_step_size"] = init_step_size

    kwargs["optim"] = get_optim_type(cfg.optim.name)

    # Arguments and keywords for the internal torch optimizer
    kwargs["optim_args"] = cfg["optim"].get("args", tuple())
    kwargs["optim_kwargs"] = cfg["optim"].get("kwargs", dict())

    return kwargs


def construct_lso_init(cfg):
    type_ = cfg.name
    args = cfg.get("args", tuple())
    kwargs = cfg.get("kwargs", dict())
    return getattr(lso.init, type_)(*args, **kwargs)


def construct_lso_search_condition(cfg):
    type_ = cfg.name
    args = cfg.get("args", tuple())
    kwargs = cfg.get("kwargs", dict())

    # Set default (min, max) stepsize to be (0, ∞)
    kwargs["min_step_size"] = kwargs.get("min_step_size", 0)
    kwargs["max_step_size"] = kwargs.get("max_step_size", torch.inf)

    return getattr(lso.search, type_)(*args, **kwargs)


def get_optim_type(name):
    if name.lower() in ("customadam", "custom_adam"):
        return CustomAdam
    else:
        return getattr(torch.optim, name)
