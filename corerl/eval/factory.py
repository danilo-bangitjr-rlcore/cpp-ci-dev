from omegaconf import DictConfig

from corerl.eval.reward import RewardEval
from corerl.eval.state import StateEval
from corerl.eval.envfield import EnvFieldEval
from corerl.eval.ibe import IBE
from corerl.eval.tde import TDE
from corerl.eval.action_gap import ActionGapEval
from corerl.eval.trace_alerts import TraceAlertsEval, UncertaintyAlertsEval
# from corerl.eval.uncertainty_alerts import UncertaintyAlertsEval
from corerl.eval.actions import ActionEval
from corerl.eval.endo_obs import EndoObsEval
from corerl.eval.ensemble import EnsembleEval
from corerl.eval.train_loss import TrainLossEval
from corerl.eval.test_loss import TestLossEval
from corerl.eval.q_estimation import QEstimation
from corerl.eval.policy_improvement import PolicyImprove
from corerl.eval.curvature import Curvature
# from corerl.eval.counterfactual import Counterfactual
from corerl.eval.base_eval import BaseEval


def init_single_evaluator(cfg: DictConfig, eval_args: dict) -> BaseEval:
    name = cfg.name
    if name == 'ibe':
        return IBE(cfg, **eval_args)
    elif name == 'tde':
        return TDE(cfg, **eval_args)
    elif name == 'reward':
        return RewardEval(cfg, **eval_args)
    elif name == 'state':
        return StateEval(cfg, **eval_args)
    elif name == 'envfield':
        return EnvFieldEval(cfg, **eval_args)
    elif name == 'action_gap':
        return ActionGapEval(cfg, **eval_args)
    elif name == 'trace_alerts':
        return TraceAlertsEval(cfg, **eval_args)
    elif name == 'uncertainty_alerts':
        return UncertaintyAlertsEval(cfg, **eval_args)
    elif name == 'endo_obs':
        return EndoObsEval(cfg, **eval_args)
    elif name == 'actions':
        return ActionEval(cfg, **eval_args)
    elif name == 'ensemble':
        return EnsembleEval(cfg, **eval_args)
    elif name == 'train_loss':
        return TrainLossEval(cfg, **eval_args)
    elif name == 'test_loss':
        return TestLossEval(cfg, **eval_args)
    elif name == 'q_estimation':
        return QEstimation(cfg, **eval_args)
    elif name == 'policy_improvement':
        return PolicyImprove(cfg, **eval_args)
    elif name == 'curvature':
        return Curvature(cfg, **eval_args)
    # elif name == 'counterfactual':
    #     return Counterfactual(cfg, **eval_args)
    else:
        raise NotImplementedError
