from typing import Any

from lib_agent.critic.adv_critic import AdvConfig, AdvCritic
from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic


def get_critic(
    cfg: dict[str, Any],
    state_dim: int,
    action_dim: int,
):
    name = cfg['name']
    if name == 'QRC':
        return QRCCritic(QRCConfig(**cfg), state_dim, action_dim)
    if name == 'Adv':
        return AdvCritic(AdvConfig(**cfg), state_dim, action_dim)
    raise NotImplementedError()
