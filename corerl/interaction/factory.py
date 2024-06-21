from omegaconf import DictConfig
import gymnasium

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.interaction.timed_interaction import TimedInteraction
from corerl.interaction.anytime_interaction import AnytimeInteraction
from corerl.interaction.offline_anytime import OfflineAnytimeInteraction
from corerl.state_constructor.base import BaseStateConstructor
from corerl.alerts.composite_alert import CompositeAlert


def init_interaction(cfg: DictConfig, env: gymnasium.Env, sc: BaseStateConstructor, alerts: CompositeAlert, **kwargs) -> BaseInteraction:
    name = cfg.name
    if name == "normalizer":
        return NormalizerInteraction(cfg, env, sc, alerts)
    if name == "timed":
        return TimedInteraction(cfg, env, sc, alerts)
    if name == 'anytime':
        return AnytimeInteraction(cfg, env, sc, alerts)
    if name == 'offline_anytime':
        data_loader = kwargs["data_loader"]
        return OfflineAnytimeInteraction(cfg, env, sc, alerts, data_loader=data_loader)
    else:
        raise NotImplementedError
