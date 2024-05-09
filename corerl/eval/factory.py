from omegaconf import DictConfig

from corerl.eval.reward import RewardEval
from corerl.eval.state import StateEval
from corerl.eval.envfield import EnvFieldEval
from corerl.eval.ibe import IBE
from corerl.eval.action_gap import ActionGapEval
from corerl.eval.base_eval import BaseEval


def init_single_evaluator(cfg: DictConfig, eval_args: dict) -> BaseEval:
    name = cfg.name
    if name == 'ibe':
        return IBE(cfg, **eval_args)
    elif name == 'reward':
        return RewardEval(cfg, **eval_args)
    elif name == 'state':
        return StateEval(cfg, **eval_args)
    elif name == 'envfield':
        return EnvFieldEval(cfg, **eval_args)
    elif name == 'action_gap':
        return ActionGapEval(cfg, **eval_args)
    else:
        raise NotImplementedError
