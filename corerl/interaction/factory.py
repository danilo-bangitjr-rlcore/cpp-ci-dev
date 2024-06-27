from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteraction
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.interaction.offline_anytime import OfflineAnytimeInteraction
from corerl.state_constructor.base import BaseStateConstructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


def init_interaction(cfg: DictConfig,
                     env: gymnasium.Env,
                     sc: BaseStateConstructor,
                     alerts: CompositeAlert,
                     transition_creator: AnytimeTransitionCreator,
                     normalizer:  ObsTransitionNormalizer,
                     **kwargs) -> BaseInteraction:
    name = cfg.name
    if name == 'anytime':
        return AnytimeInteraction(cfg, env, sc, alerts, transition_creator, normalizer)
    if name == 'offline_anytime':
        transitions = kwargs["transitions"]
        return OfflineAnytimeInteraction(cfg, env, sc, alerts,
                                         transition_creator, normalizer, transitions=transitions)
    else:
        raise NotImplementedError
