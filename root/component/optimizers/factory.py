from omegaconf import DictConfig
from typing import Optional
import torch
from root.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from root.component.optimizers.custom_torch_opts import CustomAdam

def init_optimizer(cfg: DictConfig, param: list|dict, ensemble: Optional['bool']=False):
    """
    config files: root/config/agent/critic/critic_optimizer or root/config/agent/actor/actor_optimizer
    """
    name = cfg.name
    lr = cfg.lr
    weight_decay = cfg.weight_decay

    # TODO: Han can you make sure this file is ok?
    if ensemble:
        kwargs = {
            'weight_decay': cfg.weight_decay
        }
        if name == "rms_prop":
            return EnsembleOptimizer(torch.optim.RMSprop, param, lr=lr, kwargs=kwargs)
        elif name == 'adam':
            return EnsembleOptimizer(torch.optim.Adam, param, lr=lr, kwargs=kwargs)
        elif name == 'custom_adam':
            return EnsembleOptimizer(CustomAdam, param, lr=lr, kwargs=kwargs)
        elif name == "sgd":
            return EnsembleOptimizer(torch.optim.SGD, param, lr=lr, kwargs=kwargs)
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
        else:
            raise NotImplementedError
