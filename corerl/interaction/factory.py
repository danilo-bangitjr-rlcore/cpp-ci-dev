from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteractionConfig
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.state_constructor.base import BaseStateConstructor
from corerl.data.transition_creator import BaseTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


def init_interaction(cfg: BaseInteractionConfig,
                     env: gymnasium.Env,
                     sc: BaseStateConstructor,
                     agent_tc: BaseTransitionCreator,
                     obs_normalizer:  ObsTransitionNormalizer,
                     **kwargs) -> BaseInteraction:
    name = cfg.name
    if name == 'anytime':
        return AnytimeInteraction(cfg, env, sc, obs_normalizer, agent_tc)
    else:
        raise NotImplementedError
