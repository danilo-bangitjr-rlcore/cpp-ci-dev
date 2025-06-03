from typing import Any

from ml_instrumentation.Collector import Collector

from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic


def get_critic(
    cfg: dict[str, Any],
    seed: int,
    state_dim: int,
    action_dim: int,
    collector: Collector,
):
    name = cfg['name']
    if name == 'QRC':
        return QRCCritic(QRCConfig(**cfg), seed, state_dim, action_dim, collector)

    raise NotImplementedError()
