from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.state_constructor.base import BaseStateConstructor


def init_interaction(cfg: DictConfig, env: gymnasium.Env, sc: BaseStateConstructor) -> BaseInteraction:
    name = cfg.name
    if name == "normalizer":
        return NormalizerInteraction(cfg, env, sc)
    else:
        raise NotImplementedError
