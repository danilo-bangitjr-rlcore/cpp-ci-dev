from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.interaction.offline_anytime import OfflineAnytimeInteraction
from corerl.state_constructor.base import BaseStateConstructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.transition_creator import AnytimeTransitionCreator


def init_interaction(cfg: DictConfig,
                     env: gymnasium.Env,
                     sc: BaseStateConstructor,
                     alerts: CompositeAlert,
                     transition_creator: AnytimeTransitionCreator,
                     **kwargs) -> BaseInteraction:
    name = cfg.name
    if name == "normalizer":
        return NormalizerInteraction(cfg, env, sc, alerts, transition_creator)
    if name == 'anytime':
        return AnytimeInteraction(cfg, env, sc, alerts, transition_creator)
    if name == 'offline_anytime':
        data_loader = kwargs["data_loader"]
        # TODO: add transition_creator here
        return OfflineAnytimeInteraction(cfg, env, sc, alerts, data_loader=data_loader)
    else:
        raise NotImplementedError
