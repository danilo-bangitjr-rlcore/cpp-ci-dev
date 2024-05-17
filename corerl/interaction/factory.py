from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.interaction.timed_interaction import TimedInteraction
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.state_constructor.base import BaseStateConstructor



def init_interaction(
    cfg: DictConfig,
    env: gymnasium.Env,
    sc: BaseStateConstructor,
    agent,
) -> BaseInteraction:
    name = cfg.name
    if name == "normalizer":
        return NormalizerInteraction(cfg, env, sc, agent)
    if name == "timed":
        return TimedInteraction(cfg, env, sc, agent)
    if name == 'anytime.yaml':
        return AnytimeInteraction(cfg, env, sc, agent)
    else:
        raise NotImplementedError
