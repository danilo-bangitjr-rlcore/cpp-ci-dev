from omegaconf import DictConfig

from corerl.eval.reward import RewardEval
from corerl.eval.state import StateEval
from corerl.eval.envfield import EnvFieldEval
from corerl.eval.ibe import IBE
from corerl.eval.action_gap import ActionGapEval
from corerl.eval.alerts import AlertsEval
from corerl.eval.actions import ActionEval
from corerl.eval.endo_obs import EndoObsEval
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
    elif name == 'alerts':
        return AlertsEval(cfg, **eval_args)
    elif name == 'endo_obs':
        return EndoObsEval(cfg, **eval_args)
    elif name == 'actions':
        return ActionEval(cfg, **eval_args)
    else:
        raise NotImplementedError
