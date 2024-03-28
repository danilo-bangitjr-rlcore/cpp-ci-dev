import torch
from ensemble_optimizer import EnsembleOptimizer

def init_optimizer(cfg, param, ensemble=False):
    """
    config files: root/config/agent/critic/optimizer
    """
    name = cfg.name
    lr = cfg.lr
    weight_decay = cfg.weight_decay

    # TODO: han can you make sure this file is ok?
    if ensemble:
        if name == "RMSprop":
            return EnsembleOptimizer(torch.optim.RMSprop, param, lr=lr, kwargs=kwargs)
        elif name == 'Adam':
            return EnsembleOptimizer(torch.optim.Adam, param, lr=lr, kwargs=kwargs)
        elif name == 'CustomAdam':
            return EnsembleOptimizer(CustomADAM, param, lr=lr, kwargs=kwargs)
        elif name == "SGD":
            return EnsembleOptimizer(torch.optim.SGD, param, lr=lr, kwargs=kwargs)
        else:
            raise NotImplementedError
    else:
        if name == "RMSprop":
            return torch.optim.RMSprop(param, lr, weight_decay=weight_decay)
        elif name == 'Adam':
            return torch.optim.Adam(param, lr, weight_decay=weight_decay)
        elif name == 'CustomAdam':
            return CustomADAM(param, lr) # TODO: Han I'm not sure what to do here for weight decay
        elif name == "SGD":
            return torch.optim.SGD(param, lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError