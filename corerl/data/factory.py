from omegaconf import DictConfig
import corerl.data.transition_creator as tc
from corerl.state_constructor.base import BaseStateConstructor


def init_transition_creator(cfg: DictConfig, state_constuctor: BaseStateConstructor) -> tc.BaseTransitionCreator:
    kind = cfg.transition_kind
    if kind == "anytime":
        return tc.AnytimeTransitionCreator(cfg, state_constuctor)
    elif kind == "regular_rl":
        return tc.RegularRLTransitionCreator(cfg, state_constuctor)
    else:
        raise NotImplementedError
