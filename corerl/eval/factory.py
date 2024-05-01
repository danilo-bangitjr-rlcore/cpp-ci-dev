from omegaconf import DictConfig
from corerl.eval.simple_eval import RewardEval
from corerl.eval.ibe import IBE

def init_evaluator(cfg:  DictConfig, state_dim: int, action_dim:int):
    name = cfg.name
    if name == 'ibe':
        return IBE(cfg, state_dim, action_dim)  # TODO: seems really specific ?
    elif name == 'reward':
        return RewardEval(cfg)
    else:
        raise NotImplementedError()