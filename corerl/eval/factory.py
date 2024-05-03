from omegaconf import DictConfig
from corerl.eval.base_eval import BaseEval
from corerl.eval.reward import RewardEval
from corerl.eval.ibe import IBE


def init_evaluator(cfg: DictConfig, eval_args: dict) -> BaseEval:
    name = cfg.name
    if name == 'ibe':
        return IBE(cfg, **eval_args)  # TODO: seems really specific ?
    elif name == 'reward':
        return RewardEval(cfg, **eval_args)
    else:
        raise NotImplementedError()
