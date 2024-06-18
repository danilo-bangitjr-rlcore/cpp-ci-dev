import corerl.state_constructor.examples as examples
from corerl.state_constructor.base import BaseStateConstructor

from omegaconf import DictConfig
import gymnasium


def init_state_constructor(cfg: DictConfig, env: gymnasium.Env) -> BaseStateConstructor:
    name = cfg.name
    if name == "multi_trace":
        return examples.MultiTrace(cfg, env)
    elif name == "anytime_multi_trace":
        return examples.AnytimeMultiTrace(cfg, env)
    elif name == "identity":
        return examples.Identity(cfg, env)
    elif name == 'normalize':
        return examples.Normalize(cfg, env)
    elif name == 'simple_reseau':
        return examples.SimpleReseauAnytime(cfg, env)
    elif name == 'reseau_anytime':
        return examples.ReseauAnytime(cfg, env)
    else:
        raise NotImplementedError
