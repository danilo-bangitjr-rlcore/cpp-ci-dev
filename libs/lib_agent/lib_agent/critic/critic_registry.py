from typing import Any

from lib_agent.critic.adv_critic import AdvConfig, AdvCritic
from lib_agent.critic.critic_utils import CriticConfig
from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic


def get_critic(
    cfg: dict[str, Any] | CriticConfig,
    state_dim: int,
    action_dim: int,
):
    # Convert dict to config if needed
    if isinstance(cfg, dict):
        name = cfg['name']
        if name == 'QRC':
            cfg = QRCConfig(**cfg)
        elif name == 'Adv':
            cfg = AdvConfig(**cfg)
        else:
            raise NotImplementedError(f"Unknown critic: {name}")

    # Create critic from config
    if cfg.name == 'QRC':
        assert isinstance(cfg, QRCConfig)
        return QRCCritic(cfg, state_dim, action_dim)
    if cfg.name == 'Adv':
        assert isinstance(cfg, AdvConfig)
        return AdvCritic(cfg, state_dim, action_dim)
    raise NotImplementedError(f"Unknown critic: {cfg.name}")
